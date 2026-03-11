"""Configuration for Qmini biped locomotion environment (direct)."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from qmini.robots.qmini_cfg import QMINI_CFG


@configclass
class QminiEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition: 10 joints for action, obs = base(3+3+3) + joints(10+10) + last_action(10) = 39
    action_space = 10
    observation_space = 39
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = QMINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # action scale for joint position targets
    action_scale = 0.5
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -5.0
    rew_scale_lin_vel_z = -2.0
    rew_scale_ang_vel_xy = -0.05
    rew_scale_joint_torques = -0.0001
    rew_scale_joint_acc = -2.5e-7
    rew_scale_action_rate = -0.01
    rew_scale_flat_orientation = -1.0
    # termination thresholds
    max_base_tilt = 1.0  # radians, terminate if body tilts too much
