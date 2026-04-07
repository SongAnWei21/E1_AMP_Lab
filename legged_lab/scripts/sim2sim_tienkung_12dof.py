import argparse
import os
import time

import mujoco
import mujoco.viewer  # 使用原生 viewer
import numpy as np
import onnxruntime
from pynput import keyboard

# 导入独立的手柄控制模块 (确保 gamepad_controller.py 在同级目录)
from gamepad_controller import GamepadController


class SimToSimCfg:
    """针对 TienKung_12DOF 机器人的 Sim2Sim 配置类"""
    class sim:
        sim_duration = 10000.0  # 长时间遥控测试
        num_action = 12       
        num_obs_per_step = 51 # 3+3+3+12+12+12+2+2+2 = 51
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

        # 🔴 严格按照 ONNX 真实顺序重排的 Kp (Roll -> Pitch -> Yaw -> Knee -> AnklePitch -> AnkleRoll)
        kp = np.array([
            100, 100,  # hip_roll
            100, 100,  # hip_pitch
             50,  50,  # hip_yaw
            100, 100,  # knee
             20,  20,  # ankle_pitch
             20,  20   # ankle_roll
        ])
        
        # 🔴 严格按照 ONNX 真实顺序重排的 Kd
        kd = np.array([
            5, 5,  # hip_roll
            5, 5,  # hip_pitch
            3, 3,  # hip_yaw
            5, 5,  # knee
            2, 2,  # ankle_pitch
            2, 2   # ankle_roll
        ])

    class robot:
        # 天工是全尺寸机器人，步态周期通常较长 (0.8 ~ 0.85)
        # ⚠️ 请确保这里与你天工训练配置里的 gait_cycle 完全一致！
        gait_cycle: float = 0.8  
        gait_air_ratio_l: float = 0.37
        gait_air_ratio_r: float = 0.37
        gait_phase_offset_l: float = 0.37
        gait_phase_offset_r: float = 0.87


class MujocoRunner:
    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        
        self.session = onnxruntime.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name

        # 🔴 真正的天工关节名称，严格匹配 ONNX 打印出来的顺序
        self.joint_names = [
            'hip_roll_l_joint', 'hip_roll_r_joint',       # Roll
            'hip_pitch_l_joint', 'hip_pitch_r_joint',     # Pitch
            'hip_yaw_l_joint', 'hip_yaw_r_joint',         # Yaw
            'knee_pitch_l_joint', 'knee_pitch_r_joint',   # Knee
            'ankle_pitch_l_joint', 'ankle_pitch_r_joint', # Ankle Pitch
            'ankle_roll_l_joint', 'ankle_roll_r_joint'    # Ankle Roll
        ]
        
        self.init_joint_mapping()
        self.init_variables()
        
        # 初始化手柄控制器
        self.gamepad = GamepadController(deadzone=0.15)

    def init_joint_mapping(self):
        """物理级安全机制：通过底层传动映射获取电机 ID，彻底告别猜名字"""
        self.qpos_idx = []
        self.qvel_idx = []
        self.ctrl_idx = []
        
        joint_to_actuator = {}
        for act_id in range(self.model.nu):
            if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                target_joint_id = self.model.actuator_trnid[act_id, 0]
                joint_to_actuator[target_joint_id] = act_id

        for name in self.joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1: 
                raise ValueError(f"Joint '{name}' not found in XML!")
            
            self.qpos_idx.append(self.model.jnt_qposadr[jnt_id])
            self.qvel_idx.append(self.model.jnt_dofadr[jnt_id])
            
            if jnt_id not in joint_to_actuator:
                raise ValueError(f"Fatal: 关节 '{name}' 存在，但在 XML 中没有找到驱动它的电机 (Actuator)！")
            
            self.ctrl_idx.append(joint_to_actuator[jnt_id])
            
        print("[INFO] 天工关节与电机物理映射成功！匹配 ONNX 顺序完成！")

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        
        # 🔴 严格按照 ONNX (Roll -> Pitch -> Yaw -> Knee -> AnklePitch -> AnkleRoll) 排列初始姿态
        # ⚠️ 注意：请确这里的数值与你训练天工时 init_state 里的数值完全一致！
        self.default_dof_pos = np.array([
             0.0,  0.0,  # hip_roll  (左, 右)
            -0.3, -0.3,  # hip_pitch (左, 右)
             0.0,  0.0,  # hip_yaw   (左, 右)
             0.5,  0.5,  # knee      (左, 右)
            -0.2, -0.2,  # ankle_pitch (左, 右)
             0.0,  0.0   # ankle_roll  (左, 右)
        ])
        
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])

        self.keyboard_cmd_vel = np.array([0.0, 0.0, 0.0])
        self.command_vel = np.array([0.0, 0.0, 0.0])
        
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> np.ndarray:
        self.dof_pos = np.array([self.data.qpos[i] for i in self.qpos_idx])
        self.dof_vel = np.array([self.data.qvel[i] for i in self.qvel_idx])

        quat = self.data.sensor("orientation").data 
        obs_gravity = self.quat_rotate_inverse(quat, np.array([0, 0, -1]))

        current_obs = np.concatenate([
            self.data.sensor("angular-velocity").data,
            obs_gravity,
            self.command_vel,
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel,
            self.action,
            np.sin(2 * np.pi * self.gait_phase).reshape(2,),
            np.cos(2 * np.pi * self.gait_phase).reshape(2,),
            self.phase_ratio,
        ], axis=0).astype(np.float32)

        current_obs = np.clip(current_obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = current_obs.copy()
        return self.obs_history

    def run(self) -> None:
        self.setup_keyboard_listener()
        self.listener.start()
        print("[INFO] 天工 Sim2Sim Started using Passive Viewer.")
        print("[INFO] 🎮 摇杆接管控制 | ⌨️ 松开恢复巡航 (8/2前后, 4/6横移, 7/9转向, 5急停)\n")

        target_dof_pos = self.default_dof_pos.copy()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                step_start = time.time()

                # 双模指令融合
                pad_x, pad_y, pad_yaw = self.gamepad.get_commands()
                is_pad_active = abs(pad_x) > 0 or abs(pad_y) > 0 or abs(pad_yaw) > 0
                
                if is_pad_active:
                    self.command_vel = np.array([pad_x, pad_y, pad_yaw])
                    ctrl_mode = "🎮 手柄"
                else:
                    self.command_vel = self.keyboard_cmd_vel.copy()
                    ctrl_mode = "⌨️  键盘"

                print(f"\r[{ctrl_mode}] 目标速度 -> X(前后): {self.command_vel[0]:5.2f} | Y(横移): {self.command_vel[1]:5.2f} | Yaw(转向): {self.command_vel[2]:5.2f}        ", end="", flush=True)

                # 1. 策略推理 (50Hz)
                obs = self.get_obs()
                onnx_input = {self.input_name: obs.reshape(1, -1)}
                
                raw_action = self.session.run(None, onnx_input)[0].flatten()[:12]
                self.action[:] = np.clip(raw_action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                
                target_dof_pos = self.action * self.cfg.sim.action_scale + self.default_dof_pos

                # 2. 物理步进与 PD 计算 (200Hz)
                for _ in range(self.cfg.sim.decimation):
                    cur_q = np.array([self.data.qpos[i] for i in self.qpos_idx])
                    cur_dq = np.array([self.data.qvel[i] for i in self.qvel_idx])
                    
                    tau = self.cfg.sim.kp * (target_dof_pos - cur_q) - self.cfg.sim.kd * cur_dq
                    
                    for i, act_id in enumerate(self.ctrl_idx):
                        self.data.ctrl[act_id] = tau[i]
                    
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
        self.keyboard_cmd_vel[idx] = np.clip(self.keyboard_cmd_vel[idx] + increment, -1.0, 1.0)

    def setup_keyboard_listener(self) -> None:
        def on_press(key):
            try:
                if key.char == "8": self.adjust_command_vel(0, 0.1)
                elif key.char == "2": self.adjust_command_vel(0, -0.1)
                elif key.char == "4": self.adjust_command_vel(1, -0.1)
                elif key.char == "6": self.adjust_command_vel(1, 0.1)
                elif key.char == "7": self.adjust_command_vel(2, 0.1)
                elif key.char == "9": self.adjust_command_vel(2, -0.1)
                elif key.char == "5":
                    self.keyboard_cmd_vel[:] = 0.0
            except AttributeError: pass
        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to the exported ONNX policy")
    parser.add_argument("--model", type=str, default="/home/saw/droidup/TienKung-Lab/legged_lab/assets/tienkung2_lite/mjcf/tienkung_12dof.xml", help="Path to the MJCF/XML model file")
    args = parser.parse_args()

    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model)
    runner.run()