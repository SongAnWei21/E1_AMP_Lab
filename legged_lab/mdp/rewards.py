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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import legged_lab.mdp as mdp

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
    from legged_lab.envs.e1_21dof.e1_21dof_env import E1_21DOF_Env


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# =========================================================
# Regularization Reward
# =========================================================
def ankle_torque(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)

# ==========================================
# 腰部 (Waist Yaw) 惩罚
# ==========================================
def waist_yaw_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize waist yaw joint actions.
    惩罚腰部偏航关节的动作输出，防止躯干像拨浪鼓一样左右狂扭。
    """
    # 直接使用提前缓存好的 waist_ids
    return torch.sum(torch.abs(env.action[:, env.waist_ids]), dim=1)

def waist_yaw_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the waist yaw joint.
    惩罚腰部偏航关节输出过大的力矩，收紧核心，降低能耗。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.waist_ids]), dim=1)


# ==========================================
# 肩部俯仰 (Shoulder Pitch) 惩罚
# ==========================================
def shoulder_pitch_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize shoulder pitch joint actions.
    惩罚肩部俯仰关节的动作输出，限制手臂前后摆动的幅度。
    """
    # 索引 [0] 对应 left/right_shoulder_pitch_joint
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[0], env.right_arm_ids[0]]]), dim=1)

def shoulder_pitch_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the shoulder pitch joints.
    惩罚肩部俯仰关节输出过大的力矩，防止极其生硬的机械甩臂。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, [env.left_arm_ids[0], env.right_arm_ids[0]]]), dim=1)

def shoulder_roll_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize shoulder roll joint actions.
    惩罚肩部横滚关节的动作输出，防止机器人走路时双臂过度向两侧“张开/侧平举”。
    """
    # 索引 [1] 对应 shoulder_roll_joint
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[1], env.right_arm_ids[1]]]), dim=1)


def shoulder_roll_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the shoulder roll joints.
    惩罚肩部横滚关节输出过大的力矩，降低能耗。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, [env.left_arm_ids[1], env.right_arm_ids[1]]]), dim=1)


def shoulder_yaw_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize shoulder yaw joint actions.
    惩罚肩部偏航关节的动作输出，防止手臂绕垂直轴过度“内旋或外旋”。
    """
    # 索引 [2] 对应 shoulder_yaw_joint
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[2], env.right_arm_ids[2]]]), dim=1)


def shoulder_yaw_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the shoulder yaw joints.
    惩罚肩部偏航关节输出过大的力矩。
    """
    return torch.sum(torch.square(env.robot.data.applied_torque[:, [env.left_arm_ids[2], env.right_arm_ids[2]]]), dim=1)

def elbow_action(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize elbow joint actions.
    惩罚肘部关节的动作输出，防止手臂像大风车一样疯狂甩动。
    """
    # 使用 L1 范数 (abs)，对任何微小的乱动都保持敏感
    return torch.sum(torch.abs(env.action[:, [env.left_arm_ids[3], env.right_arm_ids[3]]]), dim=1)


def elbow_torque(env: BaseEnv | E1_21DOF_Env) -> torch.Tensor:
    """
    Penalize large torques on the elbow joints.
    惩罚肘部关节输出过大的力矩，降低能耗，让手臂动作更轻柔。
    """
    # 使用 L2 范数 (square)，重点打击极端的爆发性发力
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.elbow_joint_ids]), dim=1)


def feet_y_distance(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# =========================================================
# Periodic gait-based reward function
# =========================================================
def gait_clock(phase, air_ratio, delta_t):
    # (保持原有逻辑完全不变，仅做排版)
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: BaseEnv | TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(env: BaseEnv | TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(env: BaseEnv | TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score



# def stand_still_when_zero_command(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """当平移速度指令趋近于0时，强力惩罚所有关节的运动，防止溜冰劈叉。"""
#     asset: Articulation = env.scene[asset_cfg.name]
#     joint_vel = asset.data.joint_vel
    
#     # 统一使用 env.command_generator 获取指令
#     command_norm = torch.norm(env.command_generator.command[:, :2], dim=1)
    
#     # 判断是否处于零指令状态 (只允许原地转向或者站立)
#     is_standing = command_norm < 0.1
    
#     return torch.sum(torch.square(joint_vel), dim=1) * is_standing

def action_l2_zero_command(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """当平移指令为0时，重罚任何不为0的网络动作输出 (逼迫网络彻底放松肌肉)"""
    # 计算当前平移指令大小
    command_norm = torch.norm(env.command_generator.command[:, :2], dim=1)
    
    # 如果指令小于 0.1，说明在要求原地站立
    is_standing = command_norm < 0.1
    
    # 惩罚：网络动作的平方和 * 站立标志
    return torch.sum(torch.square(env.action), dim=1) * is_standing

def stand_still_when_zero_command(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    【经典机器狗版 Stand Still】
    当平移速度指令趋近于0时，惩罚当前关节位置与默认关节位置的绝对偏差。
    强迫机器人在零指令时回到最稳定的标准对称姿态。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. 计算当前姿态与默认标准姿态的绝对偏差 (L1 范数)
    # 对应你提供的: torch.abs(self.dof_pos - self.default_dof_pos)
    pos_error = torch.abs(asset.data.joint_pos - asset.data.default_joint_pos)
    
    # 2. 获取指令并计算模长
    command_norm = torch.norm(env.command_generator.command[:, :2], dim=1)
    
    # 3. 判定是否为“零指令”状态
    is_standing = command_norm < 0.1
    
    # 4. 仅在零指令时，输出位置偏差总和作为惩罚项
    return torch.sum(pos_error, dim=1) * is_standing


def stand_still_joint_deviation_l1(
    env,
    command_name: str,  # 显式接住这个参数，满足 Isaac Lab 的严格签名校验
    command_threshold: float = 0.1,  
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize offsets from the default joint positions when ALL commands (X, Y, Yaw) are very small.
    在全指令（平移+旋转）都趋于零时，利用 L1 范数将机器人拉回默认姿态。
    """
    # 忽略传入的 command_name，直接使用你环境里独有的 command_generator
    command = env.command_generator.command
    
    # command[:, :2] 是平移 (X,Y)，command[:, 2] 是 Yaw (原地旋转角速度)
    lin_cmd_norm = torch.norm(command[:, :2], dim=1)
    ang_cmd_abs = torch.abs(command[:, 2])
    
    # 只有当线速度和角速度同时极小，才被判定为“真正需要静立”
    is_standing = (lin_cmd_norm + ang_cmd_abs) < command_threshold
    
    # 仅在需要静立时，施加偏离默认姿态的惩罚
    return mdp.joint_deviation_l1(env, asset_cfg) * is_standing

