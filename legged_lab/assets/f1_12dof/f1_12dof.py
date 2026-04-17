# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


ARMATURE_5036 = 0.01
ARMATURE_6036 = 0.01

EFFORT_LIMIT_5036 = 21
VELOCITY_LIMIT_5036 = 7.85
EFFORT_LIMIT_6036 = 36  
VELOCITY_LIMIT_6036 = 6.28


# F1 机器人模型配置 (12 DOF)
F1_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAAC_ASSET_DIR}/f1_12dof/urdf/f1_12dof.urdf", 
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "l_hip_roll_joint": 0.0,
            "l_hip_pitch_joint": -0.3,
            "l_knee_yaw_joint": 0.0,
            "l_knee_pitch_joint": 0.5,
            "l_ankle_pitch_joint": -0.25,
            "l_ankle_roll_joint": 0.0,

            "r_hip_roll_joint": 0.0,
            "r_hip_pitch_joint": -0.3,
            "r_knee_yaw_joint": 0.0,
            "r_knee_pitch_joint": 0.5,
            "r_ankle_pitch_joint": -0.25,
            "r_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_yaw_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": EFFORT_LIMIT_6036,
                ".*_hip_pitch_joint": EFFORT_LIMIT_6036,
                ".*_knee_yaw_joint": EFFORT_LIMIT_6036,
                ".*_knee_pitch_joint": EFFORT_LIMIT_6036,
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": VELOCITY_LIMIT_6036,
                ".*_hip_pitch_joint": VELOCITY_LIMIT_6036,
                ".*_knee_yaw_joint": VELOCITY_LIMIT_6036,
                ".*_knee_pitch_joint": VELOCITY_LIMIT_6036,
            },
            stiffness={
                ".*_hip_roll_joint": 100,
                ".*_hip_pitch_joint": 100,
                ".*_knee_yaw_joint": 50,
                ".*_knee_pitch_joint": 100,
            },
            damping={
                ".*_hip_roll_joint": 4,
                ".*_hip_pitch_joint": 4,
                ".*_knee_yaw_joint": 2,
                ".*_knee_pitch_joint": 4,
            },
            armature={
                ".*_hip_roll_joint": ARMATURE_6036,
                ".*_hip_pitch_joint": ARMATURE_6036,
                ".*_knee_yaw_joint": ARMATURE_6036,
                ".*_knee_pitch_joint": ARMATURE_6036,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint", 
                ".*_ankle_roll_joint"
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": EFFORT_LIMIT_5036,
                ".*_ankle_roll_joint": EFFORT_LIMIT_5036,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": VELOCITY_LIMIT_5036,
                ".*_ankle_roll_joint": VELOCITY_LIMIT_5036,
            },
            stiffness={
                ".*_ankle_pitch_joint": 20,
                ".*_ankle_roll_joint": 20,
            },
            damping={
                ".*_ankle_pitch_joint": 2, 
                ".*_ankle_roll_joint": 2,
            },
            armature={
                ".*_ankle_pitch_joint": ARMATURE_5036,
                ".*_ankle_roll_joint": ARMATURE_5036,
            },
        )
    },
)