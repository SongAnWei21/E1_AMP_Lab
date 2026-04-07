import argparse
import json
import time
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

class TxtGaitAnalyzer:
    def __init__(self, xml_path, motion_file):
        # 1. 加载 MuJoCo 模型
        print(f"[INFO] 正在加载 XML 模型: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 2. 读取并解析 TXT(JSON) 动捕数据
        print(f"[INFO] 正在加载 AMP TXT 动作文件: {motion_file}")
        with open(motion_file, 'r') as f:
            motion_data = json.load(f)
            
        self.frames = np.array(motion_data["Frames"])
        # 提取帧率，如果文件里没有，默认使用 30Hz (0.033s) —— 对应你生成时指定的 fps 30.0
        self.dt = motion_data.get("FrameDuration", 1.0/30.0)
        self.num_frames = len(self.frames)
        print(f"✅ 加载成功！共 {self.num_frames} 帧，单帧时长: {self.dt:.3f} s")

        # 3. 状态控制变量
        self.current_frame = 0
        self.need_update = True  # 标记是否需要刷新画面
        self.auto_play = False   # 是否自动播放

        # 4. 启动键盘监听器 (后台线程)
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        try:
            if key.char == 'n':  # Next (下一帧)
                self.current_frame = (self.current_frame + 1) % self.num_frames
                self.need_update = True
                self.auto_play = False
            elif key.char == 'p':  # Previous (上一帧)
                self.current_frame = (self.current_frame - 1) % self.num_frames
                self.need_update = True
                self.auto_play = False
            elif key.char == 'r':  # Reset (重置)
                self.current_frame = 0
                self.need_update = True
                self.auto_play = False
            elif key.char == ' ':  # Space (空格键切换自动播放)
                self.auto_play = not self.auto_play
        except AttributeError:
            pass

    def apply_current_frame(self):
        """将 AMP 当前帧数据解析并覆盖到物理引擎的状态中"""
        frame_data = self.frames[self.current_frame]
        
        # AMP 12 DOF TXT 一共 30 维：
        # [0:12] Joint Pos, [12:24] Joint Vel, [24:30] Foot Pos
        
        # 1. 设定 Root 位置
        # 因为 AMP 数据没有全局位置，为了不让机器人掉下去或卡在地下，我们固定它的高度 (例如 Z=0.5m)
        self.data.qpos[0:3] = [0.0, 0.0, 0.5] 

        # 2. 设定 Root 姿态
        # 固定为水平竖直状态 (四元数 wxyz)
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
        
        # 3. 提取 12 个关节的角度 [0:12] -> 放入 qpos [7:19]
        self.data.qpos[7:19] = frame_data[0:12]
        
        # 强制更新运动学状态，重新计算机器人的几何体位置
        mujoco.mj_forward(self.model, self.data)

        # 终端打印当前状态
        timestamp = self.current_frame * self.dt
        print(f"\r▶ 当前帧: [{self.current_frame:04d} / {self.num_frames - 1}] | 时间戳: {timestamp:.3f} s", end="", flush=True)

    def run(self):
        print("\n" + "="*50)
        print("🕵️  AMP 步态分析器 (原地踏步版) 已就绪！")
        print("请在终端窗口按下以下按键进行控制：")
        print("  [ N ] 键   : 下一帧 (Next Frame)")
        print("  [ P ] 键   : 上一帧 (Prev Frame)")
        print("  [ 空格 ] 键 : 播放/暂停 (Play/Pause)")
        print("  [ R ] 键   : 归零 (Reset to 0)")
        print("="*50 + "\n")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                if self.auto_play:
                    self.current_frame = (self.current_frame + 1) % self.num_frames
                    self.need_update = True

                if self.need_update:
                    self.apply_current_frame()
                    viewer.sync()
                    self.need_update = False

                # 控制循环率，防止占用过高 CPU
                time_to_sleep = self.dt - (time.time() - step_start)
                if time_to_sleep > 0 and self.auto_play:
                    time.sleep(time_to_sleep)
                elif not self.auto_play:
                    time.sleep(0.01) # 暂停状态下降低刷新率

        self.listener.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="/home/saw/droidup/TienKung-Lab/legged_lab/assets/e1_12dof/mjcf/E1_12dof.xml")
    parser.add_argument("--txt", type=str, required=True, help="Path to the AMP TXT(JSON) motion file")
    args = parser.parse_args()

    analyzer = TxtGaitAnalyzer(args.xml, args.txt)
    analyzer.run()