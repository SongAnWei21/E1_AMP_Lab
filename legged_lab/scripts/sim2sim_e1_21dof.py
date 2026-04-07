import argparse
import os
import time

import mujoco
import mujoco.viewer  # 替换为原生 viewer
import numpy as np
import onnxruntime
from pynput import keyboard

class SimToSimCfg:
    """针对 E1_21DOF 机器人的 Sim2Sim 配置类"""
    class sim:
        sim_duration = 100.0
        num_action = 21
        num_obs_per_step = 78 
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

        # PD 参数保持不变
        kp = np.array([
            100, 100, 50, 100, 20, 20,
            100, 100, 50, 100, 20, 20,
            100,
            30, 30, 30, 30,
            30, 30, 30, 30
        ])
        
        kd = np.array([
            5, 5, 3, 5, 2, 2,
            5, 5, 3, 5, 2, 2,
            5,
            2, 2, 2, 2,
            2, 2, 2, 2
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
        self.data = mujoco.MjData(self.model)
        
        # 加载 ONNX 保持不变
        self.session = onnxruntime.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name

        self.init_variables()

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        
        # 默认姿态保持不变
        self.default_dof_pos = np.array([
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
            0.0,
            0.18, 0.06, 0.06, 0.78,
            0.18, 0.06, 0.06, 0.78
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

    def get_obs(self) -> np.ndarray:
        self.dof_pos = self.data.qpos[7:7+21].copy()
        self.dof_vel = self.data.qvel[6:6+21].copy()

        quat = self.data.sensor("orientation").data # [w, x, y, z]
        obs_gravity = self.quat_rotate_inverse(quat, np.array([0, 0, -1]))

        current_obs = np.concatenate([
            self.data.sensor("angular-velocity").data,
            obs_gravity,
            self.command_vel,
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel,
            self.action,
            np.sin(2 * np.pi * self.gait_phase),
            np.cos(2 * np.pi * self.gait_phase),
            self.phase_ratio,
        ], axis=0).astype(np.float32)

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = current_obs.copy()
        return self.obs_history

    def run(self) -> None:
        self.setup_keyboard_listener()
        self.listener.start()
        print("[INFO] Sim2Sim 21DOF Started using Passive Viewer.")

        target_dof_pos = self.default_dof_pos.copy()

        # 使用 mujoco.viewer.launch_passive 替换 mujoco_viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                step_start = time.time()

                # 1. 策略推理 (50Hz 逻辑)
                obs = self.get_obs()
                onnx_input = {self.input_name: obs.reshape(1, -1)}
                self.action[:] = self.session.run(None, onnx_input)[0].flatten()[:21]
                target_dof_pos = self.action * self.cfg.sim.action_scale + self.default_dof_pos

                # 2. 物理步进与 PD 计算 (200Hz 逻辑)
                for _ in range(self.cfg.sim.decimation):
                    # 获取当前物理状态
                    cur_q = self.data.qpos[7:7+21]
                    cur_dq = self.data.qvel[6:6+21]
                    
                    # 计算 PD 力矩
                    tau = self.cfg.sim.kp * (target_dof_pos - cur_q) - self.cfg.sim.kd * cur_dq
                    
                    # 应用力矩
                    self.data.ctrl[:] = tau
                    
                    # 物理仿真
                    mujoco.mj_step(self.model, self.data)
                
                # 3. 同步 Viewer (通常在策略更新后同步渲染)
                viewer.sync()

                self.episode_length_buf += 1
                self.calculate_gait_para()

                # 保持实时率
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
        print(f"[CMD] Vel: x={self.command_vel[0]:.2f}, y={self.command_vel[1]:.2f}, yaw={self.command_vel[2]:.2f}")

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
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model)
    runner.run()