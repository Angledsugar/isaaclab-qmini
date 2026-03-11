"""Direct RL environment for Qmini biped locomotion."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .qmini_env_cfg import QminiEnvCfg


class QminiEnv(DirectRLEnv):
    cfg: QminiEnvCfg

    def __init__(self, cfg: QminiEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.last_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.last_actions.clone()
        self.last_actions = actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Joint position control: default_pos + action * scale
        targets = self.robot.data.default_joint_pos + self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target(targets)

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b,        # (num_envs, 3) base linear velocity
                self.robot.data.root_ang_vel_b,        # (num_envs, 3) base angular velocity
                self.robot.data.projected_gravity_b,   # (num_envs, 3) projected gravity
                self.joint_pos - self.robot.data.default_joint_pos,  # (num_envs, 10) relative joint pos
                self.joint_vel,                        # (num_envs, 10) joint vel
                self.last_actions,                     # (num_envs, 10) last actions
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_lin_vel_z,
            self.cfg.rew_scale_ang_vel_xy,
            self.cfg.rew_scale_joint_torques,
            self.cfg.rew_scale_joint_acc,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_flat_orientation,
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self.robot.data.applied_torque,
            self.joint_vel,
            self.last_actions,
            self.prev_actions,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if base tilts too much (check projected gravity z-component)
        projected_gravity = self.robot.data.projected_gravity_b
        base_tilt = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0))
        terminated = base_tilt > self.cfg.max_base_tilt

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Randomize joint positions around default
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos += sample_uniform(-0.2, 0.2, joint_pos.shape, joint_pos.device)
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.last_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_lin_vel_z: float,
    rew_scale_ang_vel_xy: float,
    rew_scale_joint_torques: float,
    rew_scale_joint_acc: float,
    rew_scale_action_rate: float,
    rew_scale_flat_orientation: float,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    last_actions: torch.Tensor,
    prev_actions: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # (1) Alive bonus
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # (2) Termination penalty
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # (3) Penalize vertical linear velocity
    rew_lin_vel_z = rew_scale_lin_vel_z * torch.square(root_lin_vel[:, 2])
    # (4) Penalize angular velocity in xy
    rew_ang_vel_xy = rew_scale_ang_vel_xy * torch.sum(torch.square(root_ang_vel[:, :2]), dim=1)
    # (5) Penalize joint torques
    rew_joint_torques = rew_scale_joint_torques * torch.sum(torch.square(applied_torque), dim=1)
    # (6) Penalize joint accelerations (use joint_vel as proxy)
    rew_joint_acc = rew_scale_joint_acc * torch.sum(torch.square(joint_vel), dim=1)
    # (7) Penalize action rate
    rew_action_rate = rew_scale_action_rate * torch.sum(torch.square(last_actions - prev_actions), dim=1)
    # (8) Penalize flat orientation deviation
    rew_flat_orientation = rew_scale_flat_orientation * torch.sum(
        torch.square(projected_gravity[:, :2]), dim=1
    )

    total_reward = (
        rew_alive
        + rew_termination
        + rew_lin_vel_z
        + rew_ang_vel_xy
        + rew_joint_torques
        + rew_joint_acc
        + rew_action_rate
        + rew_flat_orientation
    )
    return total_reward
