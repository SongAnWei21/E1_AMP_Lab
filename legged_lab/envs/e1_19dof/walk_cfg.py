# Copyright (c) 2021-2024, The RSL-RL Project Developers.

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
# 🔴 导入 19DOF 的资产配置
from legged_lab.assets.e1_19dof import E1_19DOF_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401

# =====================================================================
# 步态配置 (GaitCfg)
# =====================================================================
@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38    # 左腿腾空时间比例
    gait_air_ratio_r: float = 0.38    # 右腿腾空时间比例
    gait_phase_offset_l: float = 0.38 # 左腿相位偏移 (决定左腿何时开始动作)
    gait_phase_offset_r: float = 0.88 # 右腿相位偏移 (通常与左腿错开半个周期, 0.38+0.5=0.88)
    gait_cycle: float = 0.8           # 完整步态周期的持续时间 (秒)


# =====================================================================
# 奖励函数配置 (E1_19DOF_RewardCfg)
# =====================================================================
@configclass
class E1_19DOF_RewardCfg:
    # --- 任务目标 (鼓励) ---
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={"std": 0.5}) # 跟踪 XY 轴平移速度指令
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"std": 0.5})       # 跟踪 Z 轴转向(偏航)指令

    # --- 姿态与稳定性惩罚 ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)          # 惩罚 Z 轴垂直速度 
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)       # 惩罚 XY 轴旋转速度 
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, 
        weight=-2.0 , 
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}
    ) # 强烈惩罚躯干(pelvis)的倾斜，要求尽量保持直立
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0) # 惩罚整体姿态偏离水平

    # --- 动作平滑与节能惩罚 ---
    energy = RewTerm(func=mdp.energy, weight=-1e-3)                      # 惩罚总能量消耗
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)         # 惩罚关节加速度 
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)     # 惩罚相邻帧动作指令的变化率 

    # --- 安全与限制惩罚 ---
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            # 🔴 移除了 elbow_link 的接触惩罚
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*knee_link", ".*shoulder_roll_link", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) # 如果机器人摔倒(回合终止)，给予巨大惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)     # 惩罚关节角度接近物理极限

    # --- 步态约束惩罚 ---
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link"]),
        },
    )

    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
            "threshold": 100,
            "max_reward": 400,
        },
    ) 

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link"]), "threshold": 0.2},
    ) 

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"])},
    ) 

    # --- 关节偏差惩罚  ---
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*hip_yaw_joint", ".*hip_roll_joint"  
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*shoulder_roll_joint", ".*shoulder_yaw_joint"]
            )
        },
    )

    joint_deviation_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*shoulder_pitch_joint"]
            )
        },
    )

    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["waist_yaw_joint"] 
            )
        },
    )


    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*hip_pitch_joint", ".*knee_joint", ".*ankle_pitch_joint", ".*ankle_roll_joint",
                ],
            )
        },
    )

    # stand_still = RewTerm(
    #     func=mdp.stand_still_when_zero_command,
    #     weight=-0.5
    # )

    stand_still_penalty = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-2.0,  # 惩罚权重
        params={
            "command_name": "base_velocity", 
            # 平移速度的向量长度 + 原地旋转的绝对角速度 < 0.1 才触发。
            # 加了旋转角速度后，整体数值面值会变大，所以阈值要跟着放宽。
            "command_threshold": 0.12,  
        }
    )
    # ==========================================
    # 核心步态塑形 (鼓励单腿支撑与足够长的腾空时间)
    # ==========================================
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,  
        params={
            # 监听脚部的接触传感器 (根据你前面的配置，脚底连杆叫 .*ankle_roll_link)
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
            
            # 奖励封顶阈值：防止它为了刷分“金鸡独立”不放下来。
            # 通常设置为半个步态周期。你的 gait_cycle 是 0.8，所以设为 0.4 或 0.5 最佳。
            "threshold": 0.4, 
        },
    )

    # --- 周期性步态奖励 ---
    # gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02}) 
    # gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02}) 
    # gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=0.5, params={"delta_t": 0.02})

    # --- 杂项惩罚 ---
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005) 
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)  
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0) 
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)   
    

    shoulder_roll_action = RewTerm(func=mdp.shoulder_roll_action, weight=-1.0)
    shoulder_yaw_action = RewTerm(func=mdp.shoulder_yaw_action, weight=-1.0)
    shoulder_roll_torque = RewTerm(func=mdp.shoulder_roll_torque, weight=-0.0005)
    shoulder_yaw_torque = RewTerm(func=mdp.shoulder_yaw_torque, weight=-0.0005)

    waist_yaw_action = RewTerm(func=mdp.waist_yaw_action, weight=-0.001)   
    waist_yaw_torque = RewTerm(func=mdp.waist_yaw_torque, weight=-0.0005) 

    shoulder_pitch_action = RewTerm(func=mdp.shoulder_pitch_action, weight=-0.001)   
    shoulder_pitch_torque = RewTerm(func=mdp.shoulder_pitch_torque, weight=-0.0005)  

    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0) 


# =====================================================================
# 环境配置 (E1_19DOF_WalkFlatEnvCfg)
# =====================================================================
@configclass
class E1_19DOF_WalkFlatEnvCfg:
    # 更新显示用的动作文件路径
    amp_motion_files_display = [
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B4-stand_to_walk_back_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B9-walk_turn_left90_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B10-walk_turn_left45_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B11-walk_turn_left135_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B12-walk_turn_right90_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B13-walk_turn_right45_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B14-walk_turn_right135_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B22-side_step_left_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/B23-side_step_right_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/Walk-B4-Stand_to_Walk_Back_poses.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/small_forward_boy.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_visualization/walk1300fps.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_visualization/stand_100hz.txt"
                        ]
    device: str = "cuda:0"

    # --- 场景配置 ---
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0, 
        num_envs=4096,             
        env_spacing=2.5,           
        robot=E1_19DOF_CFG,        # 🔴 导入 19DOF 机器人配置
        terrain_type="generator",  
        terrain_generator=GRAVEL_TERRAINS_CFG, 
        max_init_terrain_level=5,  
        height_scanner=HeightScannerCfg(
            enable_height_scan=False, 
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  
        ),
    )

    # --- 机器人配置 ---
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,  
        critic_obs_history_length=10, 
        action_scale=0.25,            
        # 🔴 移除 .*elbow_link
        terminate_contacts_body_names=[".*knee_link", ".*shoulder_roll_link", "pelvis"], 
        feet_body_names=[".*ankle_roll_link"], 
    )
    
    reward = E1_19DOF_RewardCfg() 
    gait = GaitCfg()              

    # --- 数据标准化 ---
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0, ang_vel=1.0, projected_gravity=1.0, commands=1.0,
            joint_pos=1.0, joint_vel=1.0, actions=1.0, height_scan=1.0,
        ),
        clip_observations=100.0, 
        clip_actions=100.0,      
        height_scan_offset=0.5,
    )

    # --- 速度指令配置 ---
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0), 
        rel_standing_envs=0.2,                   # 训练站立的env比例  0.2->0.3           
        rel_heading_envs=1.0,               
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),  
            lin_vel_y=(-0.5, 0.5),  
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi) 
        ),
    )

    # --- 噪声配置 ---
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2, ang_vel=0.2, projected_gravity=0.05,
            joint_pos=0.01, joint_vel=1.5, height_scan=0.1,
        ),
    )

    # --- 域随机化 ---
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),   
                    "dynamic_friction_range": (0.4, 0.8),  
                    "restitution_range": (0.0, 0.005),     
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0), 
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5), 
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0), 
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}, 
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}), 
    )

    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


# =====================================================================
# 训练算法配置 (E1_19DOF_WalkAgentCfg)
# =====================================================================
@configclass
class E1_19DOF_WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24       
    max_iterations = 50000       
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],  
        critic_hidden_dims=[512, 256, 128], 
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO", 
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", 
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  
        rnd_cfg=None,  
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner" 
    
    # 🔴 更新名称为 19dof
    experiment_name = "e1_19dof_walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "e1_19dof_walk"
    wandb_project = "e1_19dof_walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # --- AMP参数 ---
    amp_reward_coef = 0.4
    # 🔴 更新动作捕捉数据的路径到 e1_19dof
    amp_motion_files = [
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B4-stand_to_walk_back_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B9-walk_turn_left90_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B10-walk_turn_left45_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B11-walk_turn_left135_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B12-walk_turn_right90_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B13-walk_turn_right45_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B14-walk_turn_right135_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B22-side_step_left_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/B23-side_step_right_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/Walk-B4-Stand_to_Walk_Back_poses.txt",
                        "legged_lab/envs/e1_19dof/datasets/motion_visualization/stand_100hz.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/small_forward_boy.txt",
                        # "legged_lab/envs/e1_19dof/datasets/motion_amp_expert/walk1300fps.txt"
                        ]
    amp_num_preload_transitions = 200000 
    amp_task_reward_lerp = 0.8           
    amp_discr_hidden_dims = [1024, 512, 256] 
    # 🔴 更新探索噪声的动作维度为 19
    min_normalized_std = [0.05] * 19