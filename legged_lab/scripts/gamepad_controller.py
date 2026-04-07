# 文件名: gamepad_controller.py
import pygame
import time

class GamepadController:
    def __init__(self, deadzone=0.15):
        """
        初始化手柄控制器
        :param deadzone: 摇杆死区，范围0~1，过滤摇杆物理回中的轻微抖动
        """
        # 初始化 pygame 和 joystick 模块
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.deadzone = deadzone
        
        # 检测是否连接了手柄
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[INFO] 🎮 手柄已成功连接: {self.joystick.get_name()}")
        else:
            print("[WARNING] ⚠️ 未检测到手柄，请检查无线接收器或蓝牙连接。")

    def get_commands(self):
        """
        读取手柄摇杆数据并转换为机器人的速度指令
        :return: cmd_x (前后), cmd_y (左右), cmd_yaw (转向)
        """
        # 必须调用 pump 更新 pygame 的内部事件队列，否则读不到数据
        pygame.event.pump() 
        
        if self.joystick is None:
            return 0.0, 0.0, 0.0

        # 获取摇杆轴的值 (范围 -1.0 到 1.0)
        # 注意：不同操作系统(Linux/Win)轴的映射可能略有不同，以下为标准 XInput 映射
        ly = self.joystick.get_axis(1)  # 左摇杆 Y轴 (控制前后)
        lx = self.joystick.get_axis(0)  # 左摇杆 X轴 (控制左右横移)
        
        # 针对 Linux 系统的容错处理 (某些内核下右摇杆X轴是 Axis 4 而不是 3)
        try:
            rx = self.joystick.get_axis(3)  # 右摇杆 X轴 (控制转向)
        except pygame.error:
            rx = self.joystick.get_axis(4) if self.joystick.get_numaxes() > 4 else 0.0

        # 应用死区 (Deadzone) 过滤
        ly = 0.0 if abs(ly) < self.deadzone else ly
        lx = 0.0 if abs(lx) < self.deadzone else lx
        rx = 0.0 if abs(rx) < self.deadzone else rx

        # 计算速度指令
        # 物理手柄向上推是负值，为了符合机器人坐标系(正前方为正)，需要加负号反转
        cmd_x = -ly * 1.0  # 最大前进速度 1.0 m/s
        cmd_y = -lx * 0.5  # 最大横移速度 0.5 m/s
        cmd_yaw = -rx * 1.0 # 最大自转角速度 1.0 rad/s

        return cmd_x, cmd_y, cmd_yaw


# ==========================================
# 本地测试模块 (单独运行此文件时执行)
# ==========================================
if __name__ == "__main__":
    print("--- 北通阿修罗 2 Pro 摇杆测试工具 ---")
    print("请推动摇杆，按 Ctrl+C 退出测试\n")
    
    pad = GamepadController(deadzone=0.1)
    
    if pad.joystick is not None:
        try:
            while True:
                x, y, yaw = pad.get_commands()
                # 只在摇杆有动作时打印，防止刷屏
                if abs(x) > 0 or abs(y) > 0 or abs(yaw) > 0:
                    print(f"\r[摇杆数据] X(前进): {x:5.2f} | Y(横移): {y:5.2f} | Yaw(转向): {yaw:5.2f}    ", end="")
                time.sleep(0.05) # 20Hz 采样率打印
        except KeyboardInterrupt:
            print("\n\n测试结束，退出程序。")
            pygame.quit()