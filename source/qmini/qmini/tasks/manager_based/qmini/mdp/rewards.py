"""Custom reward functions for Qmini locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def feet_symmetry(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward symmetric leg motion for bipedal gait."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    # Compare left leg (joints 0-4) with right leg (joints 5-9) symmetry
    num_joints_per_leg = joint_pos.shape[1] // 2
    left_pos = joint_pos[:, :num_joints_per_leg]
    right_pos = joint_pos[:, num_joints_per_leg:]
    return torch.sum(torch.square(left_pos - right_pos), dim=1)
