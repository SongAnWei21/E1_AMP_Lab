import argparse
import os
import time

import mujoco
import mujoco.viewer  # 使用原生 viewer
import numpy as np
import onnxruntime

# 导入独立的手柄控制模块
from gamepad_controller import GamepadController


class SimToSimCfg:
    """针对 F1_12DOF 机器人的 Sim2Sim 配置类"""
    class sim:
        sim_duration = 10000.0  # 稍微改大一点，方便长时间遥控测试
        num_action = 12       
        num_obs_per_step = 51 # 3+3+3+12+12+12+2+2+2 = 51
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

        # 🔴 严格按照 F1 ONNX 输出的顺序重排的 Kp
        kp = np.array([
            100, 100,  # 0,1: l/r_hip_roll_joint
            100, 100,  # 2,3: l/r_hip_pitch_joint
             50,  50,  # 4,5: l/r_knee_yaw_joint
            100, 100,  # 6,7: l/r_knee_pitch_joint
             20,  20,  # 8,9: l/r_ankle_pitch_joint
             20,  20   # 10,11: l/r_ankle_roll_joint
        ], dtype=np.float32)
        
        # 🔴 严格按照 F1 ONNX 输出的顺序重排的 Kd
        kd = np.array([
            4.0, 4.0,  # 0,1: l/r_hip_roll_joint
            4.0, 4.0,  # 2,3: l/r_hip_pitch_joint
            2.0, 2.0,  # 4,5: l/r_knee_yaw_joint
            4.0, 4.0,  # 6,7: l/r_knee_pitch_joint
            2.0, 2.0,  # 8,9: l/r_ankle_pitch_joint
            2.0, 2.0   # 10,11: l/r_ankle_roll_joint
        ], dtype=np.float32)

    class robot:
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

        # 🔴 从 Isaac Lab 打印出来的真实 F1 12DOF ONNX 关节顺序
        self.joint_names = [
            'l_hip_roll_joint', 'r_hip_roll_joint', 
            'l_hip_pitch_joint', 'r_hip_pitch_joint', 
            'l_knee_yaw_joint', 'r_knee_yaw_joint', 
            'l_knee_pitch_joint', 'r_knee_pitch_joint', 
            'l_ankle_pitch_joint', 'r_ankle_pitch_joint', 
            'l_ankle_roll_joint', 'r_ankle_roll_joint'
        ]
        
        self.init_joint_mapping()
        self.init_variables()
        
        # 初始化手柄控制器
        self.gamepad = GamepadController(deadzone=0.15)

    def init_joint_mapping(self):
        """物理级安全机制：通过底层传动映射获取电机 ID"""
        self.qpos_idx = []
        self.qvel_idx = []
        self.ctrl_idx = []
        
        # 建立 [目标关节 ID -> 电机 ID] 的映射字典
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
                raise ValueError(f"Fatal: 关节 '{name}' 存在，但在 XML 中没有找到驱动它的电机 (Actuator)！请检查 XML 文件。")
            
            self.ctrl_idx.append(joint_to_actuator[jnt_id])
            
        print("[INFO] F1 12DOF 关节与电机物理映射成功！告别名字错乱！")

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action, dtype=np.float32)
        self.dof_vel = np.zeros(self.cfg.sim.num_action, dtype=np.float32)
        self.action = np.zeros(self.cfg.sim.num_action, dtype=np.float32)
        
        # 🔴 严格按照 F1 ONNX 顺序排列的初始姿态 (与 config 中保持一致)
        self.default_dof_pos = np.array([
             0.0,  0.0,  # 0,1: l/r_hip_roll_joint
            -0.3, -0.3,  # 2,3: l/r_hip_pitch_joint
             0.0,  0.0,  # 4,5: l/r_knee_yaw_joint
             0.5,  0.5,  # 6,7: l/r_knee_pitch_joint
            -0.2, -0.2,  # 8,9: l/r_ankle_pitch_joint
             0.0,  0.0   # 10,11: l/r_ankle_roll_joint
        ], dtype=np.float32)
        
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2, dtype=np.float32)
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r], dtype=np.float32)
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r], dtype=np.float32)

        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> np.ndarray:
        self.dof_pos = np.array([self.data.qpos[i] for i in self.qpos_idx], dtype=np.float32)
        self.dof_vel = np.array([self.data.qvel[i] for i in self.qvel_idx], dtype=np.float32)

        quat = self.data.sensor("orientation").data # [w, x, y, z]
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

        # 裁剪观测值
        current_obs = np.clip(current_obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = current_obs.copy()
        return self.obs_history

    def run(self) -> None:
        print("[INFO] 🚀 Sim2Sim F1 12DOF Started using Passive Viewer.")
        print("[INFO] 控制说明: 仅使用手柄控制。左摇杆平移，右摇杆(左右)转向。")
        print("[INFO] 快捷指令: LT + B 退出仿真。\n")

        target_dof_pos = self.default_dof_pos.copy()

        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                    step_start = time.time()

                    # 🔴 完全按照 E1 sim2real 抄过来的摇杆读取逻辑
                    pad_x, pad_y, pad_yaw = self.gamepad.get_commands()
                    btn_b = self.gamepad.get_button_b()   
                    btn_lt = self.gamepad.get_button_lt() 

                    if btn_b and btn_lt:
                        print("\n[INFO] 🛑 接收到 LT + B，退出仿真！")
                        break

                    self.command_vel = np.array([pad_x, pad_y, pad_yaw], dtype=np.float32)

                    # 实时刷新打印当前生效的速度指令
                    print(f"\r[🎮 手柄] 目标速度 -> X(前后): {self.command_vel[0]:5.2f} | Y(横移): {self.command_vel[1]:5.2f} | Yaw(转向): {self.command_vel[2]:5.2f}        ", end="", flush=True)

                    # 1. 策略推理 (50Hz)
                    obs = self.get_obs()
                    onnx_input = {self.input_name: obs.reshape(1, -1)}
                    
                    # 获取并裁剪动作
                    raw_action = self.session.run(None, onnx_input)[0].flatten()[:12]
                    self.action[:] = np.clip(raw_action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                    
                    target_dof_pos = self.action * self.cfg.sim.action_scale + self.default_dof_pos

                    # 2. 物理步进与 PD 计算 (200Hz)
                    for _ in range(self.cfg.sim.decimation):
                        cur_q = np.array([self.data.qpos[i] for i in self.qpos_idx])
                        cur_dq = np.array([self.data.qvel[i] for i in self.qvel_idx])
                        
                        # 计算 PD 力矩
                        tau = self.cfg.sim.kp * (target_dof_pos - cur_q) - self.cfg.sim.kd * cur_dq
                        
                        # 精准地将力矩输出到对应的电机上
                        for i, act_id in enumerate(self.ctrl_idx):
                            self.data.ctrl[act_id] = tau[i]
                        
                        mujoco.mj_step(self.model, self.data)
                    
                    # 3. 渲染与状态更新
                    viewer.sync()
                    self.episode_length_buf += 1
                    self.calculate_gait_para()

                    # 4. 保持实时率
                    time_until_next_step = self.dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            print("\n[INFO] 捕捉到 Ctrl+C！退出仿真...")

    def quat_rotate_inverse(self, q, v):
        q_w = q[0]; q_vec = q[1:4]
        return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0

    def calculate_gait_para(self) -> None:
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
        self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to the exported ONNX policy")
    parser.add_argument("--model", type=str, default="/home/saw/droidup/TienKung-Lab/legged_lab/assets/f1_12dof/mjcf/f1_12dof.xml", help="Path to the MJCF/XML model file")
    args = parser.parse_args()

    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model)
    runner.run()