# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.

import argparse
import os
import time

import mujoco
import mujoco.viewer  # 原生 viewer
import numpy as np
import torch
from pynput import keyboard

class SimToSimCfg:
    class sim:
        sim_duration = 100.0
        num_action = 20       
        num_obs_per_step = 75 
        actor_obs_history_length = 10
        
        # ================== 🚨 [关键修复 1] 物理步长与频率 ==================
        dt = 0.001        # 物理引擎 1000Hz (防爆炸的核心)
        decimation = 20   # 1000Hz / 20 = 50Hz (神经网络依旧是 50Hz 控制)
        # =====================================================================
        
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25   

        kp = np.array([
            700.0, 700.0, 500.0, 700.0, 30.0, 16.8,  # 左腿
            700.0, 700.0, 500.0, 700.0, 30.0, 16.8,  # 右腿
            60.0, 20.0, 10.0, 10.0,                  # 左臂
            60.0, 20.0, 10.0, 10.0                   # 右臂
        ])
        
        kd = np.array([
            10.0, 10.0, 5.0, 10.0, 2.5, 1.4,         # 左腿
            10.0, 10.0, 5.0, 10.0, 2.5, 1.4,         # 右腿
            3.0, 1.5, 1.0, 1.0,                      # 左臂
            3.0, 1.5, 1.0, 1.0                       # 右臂
        ])

        tau_limit = np.array([
            180.0, 300.0, 180.0, 300.0, 60.0, 30.0,  # 左腿
            180.0, 300.0, 180.0, 300.0, 60.0, 30.0,  # 右腿
            52.5, 52.5, 52.5, 52.5,                  # 左臂
            52.5, 52.5, 52.5, 52.5                   # 右臂
        ])

    class robot:
        gait_cycle: float = 0.85
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88

class MujocoRunner:
    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        
        # ================== 🚨 [关键修复 2] 更换不爆引擎的积分器 ==================
        # 开启 MuJoCo 次世代隐式积分器，专治高 Kp 抽搐和开局穿模
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # =========================================================================

        self.data = mujoco.MjData(self.model)
        
        print(f"[INFO] 正在加载 PyTorch 策略模型: {policy_path}")
        self.device = torch.device("cpu") 
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.init_variables()
        self.build_joint_mappings()

    def build_joint_mappings(self):
        self.joint_names = [
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint", "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint", "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint"
        ]
        
        self.qpos_idx = np.zeros(20, dtype=int)
        self.qvel_idx = np.zeros(20, dtype=int)
        self.ctrl_idx = np.zeros(20, dtype=int)

        for i, name in enumerate(self.joint_names):
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1:
                raise ValueError(f"🚨 致命错误：在 XML 中找不到名为 '{name}' 的关节！")
            
            self.qpos_idx[i] = self.model.jnt_qposadr[jnt_id]
            self.qvel_idx[i] = self.model.jnt_dofadr[jnt_id]
            
            actuator_id = -1
            for a_id in range(self.model.nu):
                if self.model.actuator_trnid[a_id, 0] == jnt_id:
                    actuator_id = a_id
                    break
            if actuator_id == -1:
                raise ValueError(f"🚨 致命错误：在 XML 中找不到驱动 '{name}' 的电机！")
            
            self.ctrl_idx[i] = actuator_id
            
        print("[INFO] 关节映射与底层积分器配置完成。")

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        
        self.default_dof_pos = np.array([
            0.0, -0.5, 0.0, 1.0, -0.5, 0.0,   # 左腿
            0.0, -0.5, 0.0, 1.0, -0.5, 0.0,   # 右腿
            0.0, 0.1, 0.0, -0.3,              # 左臂
            0.0, -0.1, 0.0, -0.3              # 右臂
        ])
        
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])

        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> torch.Tensor:
        self.dof_pos = self.data.qpos[self.qpos_idx].copy()
        self.dof_vel = self.data.qvel[self.qvel_idx].copy()

        # ================== 🚨 [关键修复 3] 手搓安全的机体角速度 ==================
        quat = self.data.qpos[3:7].copy() # [w, x, y, z] 绝对安全的读取，不依赖 Sensor 命名
        obs_gravity = self.quat_rotate_inverse(quat, np.array([0, 0, -1]))
        
        # 提取全局角速度并转换为机体局部坐标系
        world_ang_vel = self.data.qvel[3:6].copy()
        base_ang_vel = self.quat_rotate_inverse(quat, world_ang_vel)
        # =========================================================================

        current_obs = np.concatenate([
            base_ang_vel,
            obs_gravity,
            self.command_vel,
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel,
            self.action,
            np.sin(2 * np.pi * self.gait_phase).reshape(2,),
            np.cos(2 * np.pi * self.gait_phase).reshape(2,),
            self.phase_ratio,
        ], axis=0).astype(np.float32)

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = current_obs.copy()
        
        return torch.from_numpy(self.obs_history).unsqueeze(0).to(self.device)

    def run(self) -> None:
        self.setup_keyboard_listener()
        self.listener.start()
        print("[INFO] Sim2Sim 20DOF 启动，随时准备起飞 🚀")

        target_dof_pos = self.default_dof_pos.copy()

        self.data.qpos[2] = 1.0  
        self.data.qpos[self.qpos_idx] = self.default_dof_pos.copy()
        mujoco.mj_forward(self.model, self.data)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                step_start = time.time()

                obs_tensor = self.get_obs()
                
                with torch.no_grad():
                    action_tensor = self.policy(obs_tensor)
                    
                    # 🚨 新增：防止神经网络看到没见过的环境直接吓到输出 NaN 导致引擎死锁
                    if torch.isnan(action_tensor).any():
                        print("🚨 警告：策略网络输出了 NaN，仿真紧急中止！")
                        break
                        
                self.action[:] = action_tensor.squeeze().cpu().numpy()[:20]
                target_dof_pos = self.action * self.cfg.sim.action_scale + self.default_dof_pos

                for _ in range(self.cfg.sim.decimation):
                    cur_q = self.data.qpos[self.qpos_idx]
                    cur_dq = self.data.qvel[self.qvel_idx]
                    
                    tau = self.cfg.sim.kp * (target_dof_pos - cur_q) - self.cfg.sim.kd * cur_dq
                    tau = np.clip(tau, -self.cfg.sim.tau_limit, self.cfg.sim.tau_limit)
                    
                    self.data.ctrl[self.ctrl_idx] = tau
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()

                self.episode_length_buf += 1
                self.calculate_gait_para()

                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        self.listener.stop()

    def quat_rotate_inverse(self, q, v):
        q_w = q[0]; q_vec = q[1:4]
        return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0

    def calculate_gait_para(self) -> None:
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

    def adjust_command_vel(self, idx, increment):
        self.command_vel[idx] = np.clip(self.command_vel[idx] + increment, -1.0, 1.0)
        print(f"\r[CMD] Vel: x={self.command_vel[0]:.2f}, y={self.command_vel[1]:.2f}, yaw={self.command_vel[2]:.2f}", end="")

    def setup_keyboard_listener(self) -> None:
        def on_press(key):
            try:
                if key.char == "8": self.adjust_command_vel(0, 0.1)
                elif key.char == "2": self.adjust_command_vel(0, -0.1)
                elif key.char == "4": self.adjust_command_vel(1, -0.1)
                elif key.char == "6": self.adjust_command_vel(1, 0.1)
                elif key.char == "7": self.adjust_command_vel(2, 0.1)
                elif key.char == "9": self.adjust_command_vel(2, -0.1)
            except AttributeError: pass
        self.listener = keyboard.Listener(on_press=on_press)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to policy.pt")
    parser.add_argument("--model", type=str, required=True, help="Path to robot mjcf xml")
    args = parser.parse_args()

    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model)
    runner.run()