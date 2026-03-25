"""Configuration for the Qmini biped robot.

Qmini is a bipedal robot with 10 actuated joints:
  - Left leg (LL): hip_yaw, hip_roll, hip_pitch, knee, ankle
  - Right leg (RL): hip_yaw, hip_roll, hip_pitch, knee, ankle

Joint names in URDF: LL_joint1..5, RL_joint1..5
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

QMINI_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "qmini", "urdf", "Qmini.urdf",
)

QMINI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=QMINI_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=None,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            # Left leg
            "LL_joint1": 0.0,   # hip_yaw
            "LL_joint2": 0.0,   # hip_roll
            "LL_joint3": 0.4,   # hip_pitch
            "LL_joint4": -0.8,  # knee
            "LL_joint5": 0.4,   # ankle
            # Right leg
            "RL_joint1": 0.0,   # hip_yaw
            "RL_joint2": 0.0,   # hip_roll
            "RL_joint3": 0.4,   # hip_pitch
            "RL_joint4": -0.8,  # knee
            "RL_joint5": 0.4,   # ankle
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint1"],
            effort_limit=20.0,
            velocity_limit=1.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint2"],
            effort_limit=60.0,
            velocity_limit=0.3,
            stiffness=40.0,
            damping=2.0,
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint3"],
            effort_limit=20.0,
            velocity_limit=1.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint4"],
            effort_limit=20.0,
            velocity_limit=1.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint5"],
            effort_limit=20.0,
            velocity_limit=1.0,
            stiffness=20.0,
            damping=1.0,
        ),
    },
)
"""Configuration for the Qmini biped robot.

Qmini has 10 DOF (5 per leg):
  - joint1: hip yaw  (effort 20N, vel 1 rad/s)
  - joint2: hip roll (effort 60N, vel 0.3 rad/s)
  - joint3: hip pitch (effort 20N, vel 1 rad/s)
  - joint4: knee     (effort 20N, vel 1 rad/s)
  - joint5: ankle    (effort 20N, vel 1 rad/s)

Total mass: ~9.6 kg
"""
