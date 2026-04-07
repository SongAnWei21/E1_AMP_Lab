import argparse
import os
import time

import mujoco
import mujoco.viewer  
import numpy as np
import onnxruntime
# from pynput import keyboard # 暂时注释掉，避免 Linux 下死锁

class SimToSimCfg:
    """针对 G1 12DOF 机器人的终极防弹版 Sim2Sim 配置类"""
    class sim:
        sim_duration = 100.0
        num_action = 12       
        num_obs_per_step = 51 # 必须是 51 维
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        action_scale = 0.25
        
        # 🟢 缩放系数 (必须和你的 Isaac Gym 配置文件完全一致)
        lin_vel_scale = 2.0
        ang_vel_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05

        # PD 控制参数
        kp = np.array([
            100, 100, 50, 100, 20, 20, # 左腿
            100, 100, 50, 100, 20, 20  # 右腿
        ])
        kd = np.array([
            5, 5, 3, 5, 2, 2, # 左腿
            5, 5, 3, 5, 2, 2  # 右腿
        ])

    class robot:
        # 🟢 严格定义 ONNX 模型期待的 12 个关节顺序名单
        # 必须与你训练时 env.dof_names 的顺序完全一致
        joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]

class MujocoRunner:
    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        
        self.session = onnxruntime.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name

        self.init_variables()

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        
        # ⚠️ 默认微蹲姿态 (请务必替换为你自己训练 Config 里的真实值)
        # 例如 G1 可能是这样：
        self.default_dof_pos = np.array([
            -0.3, 0.0, 0.0, 0.6, -0.3, 0.0, # 左腿
            -0.3, 0.0, 0.0, 0.6, -0.3, 0.0  # 右腿
        ]) 

        self.command_vel = np.array([0.0, 0.0, 0.0]) # [vx, vy, yaw]
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def quat_rotate_inverse(self, q, v):
        q_w = q[0]; q_vec = q[1:4]
        return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0

    def get_obs(self) -> np.ndarray:
        # 1. 读关节数据 (通过 Sensor 名字安全读取)
        for i, name in enumerate(self.cfg.robot.joint_names):
            self.dof_pos[i] = self.data.sensor(f"{name}_pos").data[0]
            self.dof_vel[i] = self.data.sensor(f"{name}_vel").data[0]

        # 2. 读 IMU 获取四元数和角速度
        quat = self.data.sensor("orientation").data # [w, x, y, z]
        base_ang_vel = self.data.sensor("angular-velocity").data 
        
        # 3. 计算重力投影 (机器人的姿态感知)
        obs_gravity = self.quat_rotate_inverse(quat, np.array([0, 0, -1]))
        
        # 4. 获取机身线速度 (将世界坐标系速度转到机身坐标系)
        base_lin_vel_world = self.data.qvel[0:3] 
        base_lin_vel_body = self.quat_rotate_inverse(quat, base_lin_vel_world)

        # 5. 严格按照 Isaac Gym 的维度顺序和 Scale 组装 51 维观测
        # 顺序: commands(3) -> ang_vel(3) -> gravity(3) -> dof_pos(12) -> dof_vel(12) -> actions(12) -> lin_vel(3) -> disturbance(3)
        current_obs = np.concatenate([
            self.command_vel * np.array([self.cfg.sim.lin_vel_scale, self.cfg.sim.lin_vel_scale, self.cfg.sim.ang_vel_scale]), 
            base_ang_vel * self.cfg.sim.ang_vel_scale, 
            obs_gravity, 
            (self.dof_pos - self.default_dof_pos) * self.cfg.sim.dof_pos_scale, 
            self.dof_vel * self.cfg.sim.dof_vel_scale, 
            self.action, 
            base_lin_vel_body * self.cfg.sim.lin_vel_scale, 
            np.array([0.0, 0.0, 0.0]) # 没有扰动给 0
        ], axis=0).astype(np.float32)

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = current_obs.copy()
        return self.obs_history

    def run(self) -> None:
        print("[INFO] Sim2Sim 12DOF Started using Passive Viewer.")

        # ============ 🟢 初始化姿态防爆 ============
        self.data.qpos[2] = 0.75  # 抬高基座，防止出生卡在地板里
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0]) # 保证身体正立
        
        # 初始化底层关节位置以防出生抽搐
        for i, name in enumerate(self.cfg.robot.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = self.default_dof_pos[i]
            
        mujoco.mj_forward(self.model, self.data) # 强制刷新物理引擎
        # =========================================

        target_dof_pos = self.default_dof_pos.copy()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                step_start = time.time()

                # 1. 策略推理 (50Hz 逻辑)
                obs = self.get_obs()
                onnx_input = {self.input_name: obs.reshape(1, -1)}

                # 截取前 12 个动作指令
                self.action[:] = self.session.run(None, onnx_input)[0].flatten()[:12]
                target_dof_pos = self.action * self.cfg.sim.action_scale + self.default_dof_pos

                # 2. 物理步进与 PD 计算 (200Hz 逻辑)
                for _ in range(self.cfg.sim.decimation):
                    
                    # 遍历写入每个电机
                    for i, name in enumerate(self.cfg.robot.joint_names):
                        cur_q = self.data.sensor(f"{name}_pos").data[0]
                        cur_dq = self.data.sensor(f"{name}_vel").data[0]
                        
                        # 计算单个电机的力矩
                        tau = self.cfg.sim.kp[i] * (target_dof_pos[i] - cur_q) - self.cfg.sim.kd[i] * cur_dq
                        
                        # 精准写入对应的电机
                        self.data.actuator(name).ctrl[0] = tau

                    # 物理仿真
                    mujoco.mj_step(self.model, self.data)
                
                # 3. 同步 Viewer
                viewer.sync()

                # 保持实时率
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model)
    runner.run()