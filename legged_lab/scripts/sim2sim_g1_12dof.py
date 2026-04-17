import argparse
import os
import time

import mujoco
import mujoco.viewer  
import numpy as np
import onnxruntime
from pynput import keyboard

# 导入独立的手柄控制模块
from gamepad_controller import GamepadController

class SimToSimCfg:
    """针对 G1 12DOF 机器人的终极防弹版 Sim2Sim 配置类"""
    class sim:
        sim_duration = 10000.0
        num_action = 12       
        num_obs_per_step = 51 # 必须是 51 维 (G1 专属)
        actor_obs_history_length = 10
        dt = 0.005
        decimation = 4
        action_scale = 0.25
        clip_actions = 100.0
        
        # 🟢 G1 专属：观测缩放系数
        lin_vel_scale = 2.0
        ang_vel_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05

        # 🔴 核心修复：Kp 和 Kd 必须跟随 ONNX 的“左右交替”顺序重排！
        kp = np.array([
            100, 100,  # hip_pitch (左, 右)
            100, 100,  # hip_roll  (左, 右)
             50,  50,  # hip_yaw   (左, 右)
            100, 100,  # knee      (左, 右)
             20,  20,  # ankle_pitch (左, 右)
             20,  20   # ankle_roll  (左, 右)
        ])
        
        kd = np.array([
            5, 5,  # hip_pitch
            5, 5,  # hip_roll
            3, 3,  # hip_yaw
            5, 5,  # knee
            2, 2,  # ankle_pitch
            2, 2   # ankle_roll
        ])

    class robot:
        # 🔴 核心修复：严格按照 ONNX 的真实顺序名单
        joint_names = [
            'left_hip_pitch_joint', 'right_hip_pitch_joint', 
            'left_hip_roll_joint', 'right_hip_roll_joint', 
            'left_hip_yaw_joint', 'right_hip_yaw_joint', 
            'left_knee_joint', 'right_knee_joint', 
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
            'left_ankle_roll_joint', 'right_ankle_roll_joint'
        ]

class MujocoRunner:
    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(model_path)

        
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        
        self.session = onnxruntime.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name

        self.init_joint_mapping()
        self.init_variables()
        
        # 初始化手柄控制器
        self.gamepad = GamepadController(deadzone=0.15)

    def init_joint_mapping(self):
        """物理级安全机制：引入防爆映射逻辑与转子惯量注入"""
        self.qpos_idx = []
        self.qvel_idx = []
        self.ctrl_idx = []
        
        joint_to_actuator = {}
        for act_id in range(self.model.nu):
            if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                target_joint_id = self.model.actuator_trnid[act_id, 0]
                joint_to_actuator[target_joint_id] = act_id

        for name in self.cfg.robot.joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id == -1: 
                raise ValueError(f"Joint '{name}' not found in XML!")
            
            self.qpos_idx.append(self.model.jnt_qposadr[jnt_id])
            self.qvel_idx.append(self.model.jnt_dofadr[jnt_id])
            
            if jnt_id not in joint_to_actuator:
                raise ValueError(f"Fatal: 关节 '{name}' 存在，但在 XML 中没有找到驱动它的电机！")
            
            self.ctrl_idx.append(joint_to_actuator[jnt_id])
            
        # ====== 🚨 核心抢救：强行注入转子惯量 (Armature) ======
        # 严格按照 G1 的 ONNX 顺序：[pitch, roll, yaw, knee, ankle_pitch, ankle_roll]
        # armatures = [
        #     0.02,  0.02,   # l/r_hip_pitch
        #     0.02,  0.02,   # l/r_hip_roll
        #     0.015, 0.015,  # l/r_hip_yaw
        #     0.02,  0.02,   # l/r_knee
        #     0.01,  0.01,   # l/r_ankle_pitch
        #     0.005, 0.005   # l/r_ankle_roll
        # ]
        # armatures = [
        #     0.0,  0.0,   # l/r_hip_pitch
        #     0.0,  0.0,   # l/r_hip_roll
        #     0.0,  0.0,  # l/r_hip_yaw
        #     0.0,  0.0,   # l/r_knee
        #     0.01,  0.01,   # l/r_ankle_pitch
        #     0.005, 0.005   # l/r_ankle_roll
        # ]
        # for i, idx in enumerate(self.qvel_idx):
        #     self.model.dof_armature[idx] = armatures[i]

        # print("[INFO] G1 关节与电机物理映射成功！已注入转子惯量！")

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        
        # 🔴 核心修复：初始姿态必须跟随 ONNX 顺序重排！
        self.default_dof_pos = np.array([
            -0.3, -0.3,  # hip_pitch (左, 右)
             0.0,  0.0,  # hip_roll  (左, 右)
             0.0,  0.0,  # hip_yaw   (左, 右)
             0.6,  0.6,  # knee      (左, 右)
            -0.3, -0.3,  # ankle_pitch (左, 右)
             0.0,  0.0   # ankle_roll  (左, 右)
        ]) 

        self.keyboard_cmd_vel = np.array([0.0, 0.0, 0.0]) # [vx, vy, yaw]
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def quat_rotate_inverse(self, q, v):
        q_w = q[0]; q_vec = q[1:4]
        return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0

    def get_obs(self) -> np.ndarray:
        # 快速读取关节数据 (此时读取的顺序已经完美对齐 ONNX)
        self.dof_pos = np.array([self.data.qpos[i] for i in self.qpos_idx])
        self.dof_vel = np.array([self.data.qvel[i] for i in self.qvel_idx])

        # 读 IMU 获取四元数和角速度
        quat = self.data.sensor("orientation").data # [w, x, y, z]
        base_ang_vel = self.data.sensor("angular-velocity").data 
        
        # 计算重力投影
        obs_gravity = self.quat_rotate_inverse(quat, np.array([0, 0, -1]))
        
        # 获取机身线速度
        base_lin_vel_world = self.data.qvel[0:3] 
        base_lin_vel_body = self.quat_rotate_inverse(quat, base_lin_vel_world)

        # G1 的 51 维组装逻辑
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
        self.setup_keyboard_listener()
        self.listener.start()
        print("[INFO] G1 Sim2Sim Started using Passive Viewer.")
        print("[INFO] 🎮 摇杆接管控制 | ⌨️ 松开恢复巡航 (8/2前后, 4/6横移, 7/9转向, 5急停)\n")

        # ============ G1 初始化姿态防爆 ============
        self.data.qpos[2] = 0.75  # 抬高基座，防止出生卡在地板里
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0]) # 保证身体正立
        
        for i, qpos_id in enumerate(self.qpos_idx):
            self.data.qpos[qpos_id] = self.default_dof_pos[i]
            
        mujoco.mj_forward(self.model, self.data) # 强制刷新物理引擎
        # =========================================

        target_dof_pos = self.default_dof_pos.copy()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.data.time < self.cfg.sim.sim_duration:
                step_start = time.time()

                pad_x, pad_y, pad_yaw = self.gamepad.get_commands()
                is_pad_active = abs(pad_x) > 0 or abs(pad_y) > 0 or abs(pad_yaw) > 0
                
                if is_pad_active:
                    self.command_vel = np.array([pad_x, pad_y, pad_yaw])
                    ctrl_mode = "🎮 手柄"
                else:
                    self.command_vel = self.keyboard_cmd_vel.copy()
                    ctrl_mode = "⌨️  键盘"

                print(f"\r[{ctrl_mode}] 目标速度 -> X: {self.command_vel[0]:5.2f} | Y: {self.command_vel[1]:5.2f} | Yaw: {self.command_vel[2]:5.2f}        ", end="", flush=True)

                # 1. 策略推理 (50Hz)
                obs = self.get_obs()
                onnx_input = {self.input_name: obs.reshape(1, -1)}

                raw_action = self.session.run(None, onnx_input)[0].flatten()[:12]
                raw_action = np.clip(raw_action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                
                # 智能平滑手刹
                if np.linalg.norm(self.command_vel) < 0.01:
                    self.action[:] = 0.05 * raw_action + 0.95 * self.action
                else:
                    self.action[:] = 0.4 * raw_action + 0.6 * self.action

                target_dof_pos = self.action * 0 + self.default_dof_pos

                # 2. 物理步进与 PD 计算 (200Hz)
                for _ in range(self.cfg.sim.decimation):
                    cur_q = np.array([self.data.qpos[i] for i in self.qpos_idx])
                    cur_dq = np.array([self.data.qvel[i] for i in self.qvel_idx])
                    
                    tau = self.cfg.sim.kp * (target_dof_pos - cur_q) - self.cfg.sim.kd * cur_dq
                    
                    for i, act_id in enumerate(self.ctrl_idx):
                        self.data.ctrl[act_id] = tau[i]
                    
                    mujoco.mj_step(self.model, self.data)
                
                # 3. 渲染与状态更新
                viewer.sync()

                # 4. 保持实时率
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        self.listener.stop()

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
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/saw/droidup/TienKung-Lab/legged_lab/assets/g1_12dof/g1_12dof.xml", help="Path to the MJCF/XML model file")
    args = parser.parse_args()

    sim_cfg = SimToSimCfg()
    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model)
    runner.run()