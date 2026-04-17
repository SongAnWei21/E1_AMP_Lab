import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
import pickle

# ================= 配置区域 (12-DoF 纯腿部) =================
# 仅保留双腿的 12 个关节
XML_JOINT_NAMES = [
    # Left Leg (6)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    
    # Right Leg (6)
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
]
# ==========================================================

def play_motion(xml_path, motion_file):
    print(f"Loading XML: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    print(f"Loading PKL Motion: {motion_file}")
    with open(motion_file, 'rb') as f:
        motion = pickle.load(f)
    
    # 1. 读取关节数据
    if 'dof_pos' in motion:
        pkl_qpos = motion['dof_pos'] # Shape: (Frames, Num_Joints)
    elif 'joint_pos' in motion:
        pkl_qpos = motion['joint_pos']
    elif 'qpos' in motion:
        pkl_qpos = motion['qpos']
    else:
        print("Error: Cannot find 'dof_pos' or 'joint_pos' in PKL file.")
        print(f"Available keys: {list(motion.keys())}")
        return

    # 2. 读取 Root 位置/姿态
    pkl_root_pos = motion.get('root_pos', motion.get('body_pos_w', None))
    pkl_root_quat_xyzw = motion.get('root_rot', motion.get('body_quat_w', None))

    # 3. 核心修复：四元数格式转换 (PKL 的 xyzw -> MuJoCo 的 wxyz)
    if pkl_root_quat_xyzw is not None:
        pkl_root_quat = np.zeros_like(pkl_root_quat_xyzw)
        pkl_root_quat[..., 0] = pkl_root_quat_xyzw[..., 3]  # w
        pkl_root_quat[..., 1] = pkl_root_quat_xyzw[..., 0]  # x
        pkl_root_quat[..., 2] = pkl_root_quat_xyzw[..., 1]  # y
        pkl_root_quat[..., 3] = pkl_root_quat_xyzw[..., 2]  # z
    else:
        pkl_root_quat = None

    # 4. 检查维度，确保数据帧和机器人模型匹配
    num_frames = pkl_qpos.shape[0]
    num_joints_pkl = pkl_qpos.shape[1]
    num_joints_xml = len(XML_JOINT_NAMES)

    print(f"Frames: {num_frames}, PKL Joints: {num_joints_pkl}, XML Joints Expected: {num_joints_xml}")

    if num_joints_pkl != num_joints_xml:
        print(f"[WARNING] Joint count mismatch! PKL has {num_joints_pkl}, but script expects {num_joints_xml}.")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("\nStarting Playback (12-DoF)...")
        frame_idx = 0
        
        while viewer.is_running():
            step_start = time.time()

            # --- 实时打印帧数与进度 ---
            progress = (frame_idx + 1) / num_frames * 100
            print(f"\r[Playback] Frame: {frame_idx + 1:04d} / {num_frames} | Progress: {progress:6.2f}%", end="", flush=True)

            # 1. 设置 Root (Base) 位置
            if pkl_root_pos is not None:
                if pkl_root_pos.ndim == 3: 
                    d.qpos[0:3] = pkl_root_pos[frame_idx, 0, :]
                else:
                    d.qpos[0:3] = pkl_root_pos[frame_idx]

            # 2. 设置 Root 姿态 (已经转换为 wxyz)
            if pkl_root_quat is not None:
                if pkl_root_quat.ndim == 3:
                    d.qpos[3:7] = pkl_root_quat[frame_idx, 0, :]
                else:
                    d.qpos[3:7] = pkl_root_quat[frame_idx]
            
            # 3. 设置关节角度 (截取前 num_joints_xml 个，防止 PKL 是 21DoF 的)
            current_frame_joints = pkl_qpos[frame_idx, :num_joints_xml] 
            d.qpos[7 : 7+num_joints_xml] = current_frame_joints

            # 4. 刷新运动学并更新画面
            mujoco.mj_forward(m, d)
            viewer.sync()

            # 循环播放逻辑
            frame_idx = (frame_idx + 1) % num_frames
            if frame_idx == 0:
                print("\n[Info] Loop finished, restarting...")
            
            # 控制播放速度 (0.02s per frame 约等于 50Hz)
            time_until_next_step = 0.02 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 记得将这里的 default 改成你 12dof 机器人的 xml 路径
    parser.add_argument("--xml", type=str, default="/home/saw/droidup/TienKung-Lab/legged_lab/assets/e1_12dof/mjcf/E1_12dof.xml")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the PKL motion file")
    args = parser.parse_args()

    play_motion(args.xml, args.pkl)