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
from legged_lab.assets.e1_21dof import E1_21DOF_CFG
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
# 定义机器人行走时的相位和周期。
# 这些参数用于生成供神经网络参考的相位信息（如 sin/cos 相位）。
# =====================================================================
@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.4    # 左腿腾空时间比例
    gait_air_ratio_r: float = 0.4    # 右腿腾空时间比例
    gait_phase_offset_l: float = 0.4 # 左腿相位偏移 (决定左腿何时开始动作)
    gait_phase_offset_r: float = 0.9 # 右腿相位偏移 (通常与左腿错开半个周期, 0.38+0.5=0.88)
    gait_cycle: float = 0.7          # 完整步态周期的持续时间 (秒)


# =====================================================================
# 奖励函数配置 (E1_21DOF_RewardCfg)
# 定义强化学习训练中机器人的“目标”和“惩罚”。
# =====================================================================
@configclass
class E1_21DOF_RewardCfg:
    # --- 任务目标 (鼓励) ---
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.5}) # 跟踪 XY 轴平移速度指令
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})       # 跟踪 Z 轴转向(偏航)指令

    # --- 姿态与稳定性惩罚 ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)          # 惩罚 Z 轴垂直速度 (防止机器人像兔子一样上下乱跳)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)       # 惩罚 XY 轴旋转速度 (防止躯干剧烈摇晃/翻滚)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, 
        weight=-2.0 , 
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}
    ) # 强烈惩罚躯干(pelvis)的倾斜，要求尽量保持直立
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0) # 惩罚整体姿态偏离水平

    # --- 动作平滑与节能惩罚 ---
    energy = RewTerm(func=mdp.energy, weight=-1e-5)                     # 惩罚总能量消耗
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)         # 惩罚关节加速度 (让动作更平滑)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)     # 惩罚相邻帧动作指令的变化率 (防止高频震荡)

    # --- 安全与限制惩罚 ---
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            # 惩罚膝盖、肩膀、手肘、骨盆与地面的接触 (只有脚能碰地)
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*knee_link", ".*shoulder_roll_link", ".*elbow_link", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) # 如果机器人摔倒(回合终止)，给予巨大惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)     # 惩罚关节角度接近物理极限

    # --- 步态约束惩罚 ---
    # 惩罚脚部在接触地面时的滑动 (要求踏实)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link"]),
        },
    )

    # 惩罚脚部受到过大的冲击力
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"]),
            "threshold": 100,
            "max_reward": 400,
        },
    ) 

    # 惩罚两脚距离过近 (防止绊倒)
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link"]), "threshold": 0.2},
    ) 

    # 惩罚脚部绊到障碍物
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll_link"])},
    ) 

    # --- 关节偏差惩罚  ---
    # 限制某些关节的动作幅度，使其尽量保持在默认位置附近
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
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*shoulder_pitch_joint"]
            )
        },
    )

    # 限制腰部关节的过度扭动，收紧核心，逼迫它用摆臂来抵消角动量
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-100.0,  # 给一个中等强度的负惩罚
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["waist_yaw_joint"] # 惩罚你的腰部偏航关节
            )
        },
    )
    
    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*elbow_joint"]
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

    # --- 周期性步态奖励 (强制形成周期性迈步) ---
    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=2.0, params={"delta_t": 0.02}) # 脚部接触力的周期性
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=2.0, params={"delta_t": 0.02}) # 脚部速度的周期性
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=1.5, params={"delta_t": 0.02})

    # --- 杂项惩罚 ---
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005) # 惩罚踝关节输出大扭矩
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)  # 惩罚频繁使用踝关节动作
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-1.0) # 强烈惩罚髋关节横滚(劈叉动作)
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-1.0)   # 强烈惩罚髋关节偏航(内八/外八)
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0) # 惩罚两脚在 Y 轴(横向)上的距离偏差


# =====================================================================
# 环境配置 (E1_21DOF_WalkFlatEnvCfg)
# 定义物理仿真环境、机器人属性、观测值缩放、指令范围等。
# =====================================================================
@configclass
class E1_21DOF_WalkFlatEnvCfg:
    # 纯显示用的 AMP 动作文件
    amp_motion_files_display = ["legged_lab/envs/e1_21dof/datasets/motion_visualization/small_forward_boy.txt"]
    device: str = "cuda:0"

    # --- 场景配置 ---
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0, # 每个回合最大时长 20 秒
        num_envs=4096,             # 并行训练 4096 个环境
        env_spacing=2.5,           # 每个环境间隔 2.5 米
        robot=E1_21DOF_CFG,        # 导入 E1 机器人模型配置
        terrain_type="generator",  # 地形生成器
        terrain_generator=GRAVEL_TERRAINS_CFG, # 使用碎石/崎岖地形
        max_init_terrain_level=5,  # 初始化的最高地形难度等级
        height_scanner=HeightScannerCfg(
            enable_height_scan=False, # 关闭高度扫描(地形感知)，这通常意味着在训练盲走策略
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  
        ),
    )

    # --- 机器人配置 ---
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,  # 策略网络观测历史长度
        critic_obs_history_length=10, # 价值网络观测历史长度
        action_scale=0.25,            # 神经网络输出动作的缩放比例
        terminate_contacts_body_names=[".*knee_link", ".*shoulder_roll_link", ".*elbow_link", "pelvis"], # 这些部位碰地则回合失败
        feet_body_names=[".*ankle_roll_link"], # 指定脚部连杆
    )
    
    reward = E1_21DOF_RewardCfg() # 挂载上面定义的奖励
    gait = GaitCfg()              # 挂载步态参数

    # --- 数据标准化 ---
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0, ang_vel=1.0, projected_gravity=1.0, commands=1.0,
            joint_pos=1.0, joint_vel=1.0, actions=1.0, height_scan=1.0,
        ),
        clip_observations=100.0, # 裁剪观测值极值
        clip_actions=100.0,      # 裁剪动作极值
        height_scan_offset=0.5,
    )

    # --- 速度指令配置 ---
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0), # 每 10 秒重新采样一次目标指令
        rel_standing_envs=0.2,              # 20% 的机器人被要求原地站立 (用于学习静止平衡)
        rel_heading_envs=1.0,               # 100% 的机器人使用朝向控制
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),  # 前后速度范围 -0.6 ~ 1.0 m/s
            lin_vel_y=(-0.5, 0.5),  # 左右横移速度范围
            ang_vel_z=(-1.57, 1.57),# 偏航角速度范围
            heading=(-math.pi, math.pi) # 绝对朝向范围
        ),
    )

    # --- 噪声配置 (提高策略鲁棒性) ---
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2, ang_vel=0.2, projected_gravity=0.05,
            joint_pos=0.01, joint_vel=1.5, height_scan=0.1,
        ),
    )

    # --- 域随机化 (Domain Randomization) ---
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),   # 随机静态摩擦力
                    "dynamic_friction_range": (0.4, 0.8),  # 随机动态摩擦力
                    "restitution_range": (0.0, 0.005),     # 随机恢复系数(弹性)
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0), # 随机给躯干增加 -5 到 5 kg的质量
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    # 回合重置时，给机器人一个随机的初始位置和速度
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
                    "position_range": (0.5, 1.5), # 重置时关节位置加点随机缩放
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0), # 每隔 10-15 秒推一下机器人
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}, # 模拟被人踹了一脚
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}), # 关闭动作延迟模拟
    )

    # 仿真步长 5ms，每 4 步(20ms, 50Hz)执行一次策略
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


# =====================================================================
# 训练算法配置 (E1_21DOF_WalkAgentCfg)
# 定义 PPO 强化学习算法的超参数，以及 AMP (对抗运动先验) 的设置。
# =====================================================================
@configclass
class E1_21DOF_WalkAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24       # 每次收集经验步数
    max_iterations = 50000       # 最大训练迭代次数
    empirical_normalization = False

    # --- 神经网络结构 ---
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],  # 策略网络 (Actor) 结构
        critic_hidden_dims=[512, 256, 128], # 价值网络 (Critic) 结构
        activation="elu",
    )

    # --- PPO 算法超参数 ---
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO", # 使用支持 AMP 的 PPO 算法
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", # 自适应学习率
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
    experiment_name = "e1_21dof_walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "e1_21dof_walk"
    wandb_project = "e1_21dof_walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # --- AMP参数 ---
    amp_reward_coef = 0.3 # 动作模仿奖励的权重 (判别器认为动作越真，奖励越高)
    amp_motion_files = ["legged_lab/envs/e1_21dof/datasets/motion_amp_expert/small_forward_boy.txt"] # 专家动作捕捉数据文件
    amp_num_preload_transitions = 200000 # 预加载的动捕数据帧数
    amp_task_reward_lerp = 0.7           # 任务奖励(Task Reward)和模仿奖励(AMP Reward)的融合比例
    amp_discr_hidden_dims = [1024, 512, 256] # 判别器(Discriminator)的网络结构
    min_normalized_std = [0.05] * 21     # 动作探索噪声的最小下限