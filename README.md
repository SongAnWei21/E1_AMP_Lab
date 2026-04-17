# Legged Lab - 腿式机器人强化学习框架

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Isaac Lab](https://img.shields.io/badge/Built%20on-Isaac%20Lab-orange)](https://isaaclab.github.io/)
[![RSL-RL](https://img.shields.io/badge/Based%20on-RSL--RL-green)](https://github.com/leggedrobotics/rsl_rl)

**Legged Lab** 是一个基于 [Isaac Lab](https://isaaclab.github.io/) 和 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 的腿式机器人强化学习框架，专注于为全尺寸人形机器人和其他腿式机器人提供高效、稳定、自然的运动控制。本框架集成了对抗运动先验（AMP）技术，支持从仿真到仿真的快速部署。

## ✨ 特性

- **多机器人支持**: 支持多种腿式机器人模型，包括 E1 系列（12/13/19/21自由度）、TienKung 系列、G1 和 F1 机器人
- **强化学习算法**: 基于 RSL-RL 的高效并行强化学习训练，集成 AMP（对抗运动先验）技术
- **仿真环境**: 基于 Isaac Lab 的高性能物理仿真，支持 GPU 加速的大规模并行环境
- **快速部署**: 提供 Sim2Sim（仿真到仿真）脚本，可将训练策略快速部署到 Mujoco 等物理引擎
- **传感器集成**: 支持摄像头、LiDAR 等多种传感器，用于感知环境
- **运动数据**: 提供预处理的运动捕捉数据，用于 AMP 训练
- **预训练策略**: 提供多种机器人的预训练策略，开箱即用
- **游戏手柄控制**: 支持使用游戏手柄实时控制机器人运动

## 🚀 快速开始

### 安装

1. **克隆仓库**
```bash
git clone https://github.com/TienKung-Lab/LeggedLab.git
cd LeggedLab
```

2. **安装依赖**
```bash
pip install -e .
```

3. **安装 Isaac Lab**
请参考 [Isaac Lab 官方文档](https://isaaclab.github.io/installation.html) 安装 Isaac Lab。

4. **安装 Mujoco**（用于 Sim2Sim）
```bash
pip install mujoco==3.3.2 mujoco-python-viewer
```

### 训练机器人

训练 E1 21自由度机器人行走：
```bash
python legged_lab/scripts/train.py --task=e1_21dof_walk
```

训练 TienKung 机器人跑步（带传感器）：
```bash
python legged_lab/scripts/train.py --task=run_with_sensor
```

### 仿真到仿真（Sim2Sim）

使用预训练策略在 Mujoco 中运行 E1 19自由度机器人：
```bash
python legged_lab/scripts/sim2sim_e1_19dof.py --policy=policy/e1_19dof/policy.onnx
```

使用游戏手柄控制：
- 左摇杆：前后/左右移动
- 右摇杆：左右转向
- LT + B：退出仿真

### 播放 AMP 动画

可视化训练好的策略：
```bash
python legged_lab/scripts/play_amp_animation.py --task=e1_21dof_walk
```

## 🤖 支持的机器人

| 机器人 | 自由度 | 支持任务 | 描述 |
|--------|--------|----------|------|
| **TienKung** | 全自由度 | walk, run, walk_with_sensor, run_with_sensor | 全尺寸人形机器人 |
| **TienKung 12DOF** | 12 | walk | 简化版 TienKung |
| **E1 21DOF** | 21 | walk, run | 21自由度人形机器人 |
| **E1 19DOF** | 19 | walk | 19自由度人形机器人（无肘部） |
| **E1 13DOF** | 13 | walk | 13自由度人形机器人 |
| **E1 12DOF** | 12 | walk, run | 12自由度人形机器人 |
| **G1 12DOF** | 12 | walk | 四足机器人 |
| **F1 12DOF** | 12 | walk | 另一款四足机器人 |

## 📁 项目结构

```
LeggedLab/
├── legged_lab/              # 核心框架
│   ├── assets/             # 机器人资产 (URDF/MJCF/USD)
│   │   ├── e1_12dof/      # E1 12自由度模型
│   │   ├── e1_21dof/      # E1 21自由度模型
│   │   ├── tienkung2_lite/ # TienKung 模型
│   │   └── ...
│   ├── envs/              # 强化学习环境
│   │   ├── base/          # 基础环境类
│   │   ├── e1_12dof/      # E1 12自由度环境
│   │   ├── tienkung/      # TienKung 环境
│   │   └── ...
│   ├── mdp/               # MDP 定义 (奖励、终止条件等)
│   ├── sensors/           # 传感器模块
│   ├── scripts/           # 训练和评估脚本
│   └── utils/             # 工具函数
├── rsl_rl/                # RSL-RL 强化学习库
├── policy/                # 预训练策略
│   ├── e1_12dof/         # E1 12自由度策略
│   ├── e1_21dof/         # E1 21自由度策略
│   └── ...
├── motion/               # 运动捕捉数据
│   ├── e1_12dof/        # E1 12自由度运动数据
│   └── e1_21dof/        # E1 21自由度运动数据
├── docs/                 # 文档和演示
└── setup.py             # Python 包配置
```

## 📊 训练与评估

### 训练配置

训练配置位于各环境的 `*_cfg.py` 文件中，例如：
- `legged_lab/envs/e1_21dof/walk_cfg.py` - E1 21DOF 行走配置
- `legged_lab/envs/tienkung/run_cfg.py` - TienKung 跑步配置

### 策略导出

训练完成后，策略会自动导出为 ONNX 格式，保存在 `policy/` 目录下，可用于 Sim2Sim 部署。

### 性能评估

框架提供多种评估工具：
- `play_amp_animation.py` - 可视化策略性能
- Sim2Sim 脚本 - 在 Mujoco 中测试策略
- 游戏手柄控制 - 实时交互测试

## 🔧 自定义与扩展

### 添加新机器人

1. 在 `legged_lab/assets/` 中添加机器人资产（URDF/MJCF/USD）
2. 在 `legged_lab/envs/` 中创建新环境类，继承 `BaseEnv`
3. 在 `legged_lab/envs/__init__.py` 中注册新任务
4. 创建训练配置文件（`*_cfg.py`）

### 添加新任务

1. 在现有环境目录中创建新的配置文件
2. 定义新的奖励函数和终止条件（在 `mdp/` 中）
3. 在 `legged_lab/envs/__init__.py` 中注册新任务

### 使用自定义传感器

框架支持多种传感器：
- 摄像头：RGB、深度、实例分割
- LiDAR：3D 点云
- IMU：姿态、角速度

参考 `legged_lab/sensors/` 中的示例添加新传感器。

## 📚 文档

- [Isaac Lab 文档](https://isaaclab.github.io/) - 基础仿真框架
- [RSL-RL 文档](https://github.com/leggedrobotics/rsl_rl) - 强化学习算法
- [Mujoco 文档](https://mujoco.readthedocs.io/) - 物理仿真

## 🤝 贡献

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 使用 [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) 编写文档字符串
- 使用 [black](https://black.readthedocs.io/) 代码格式化器
- 使用 [flake8](https://flake8.pycqa.org/) 代码检查

安装 pre-commit 钩子：
```bash
pre-commit install
pre-commit run --all-files
```

## 📄 引用

如果您在研究中使用了本框架，请引用以下相关文献：

### RSL-RL
```bibtex
@InProceedings{rudin2022learning,
  title = {Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author = {Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  booktitle = {Proceedings of the 5th Conference on Robot Learning},
  pages = {91--100},
  year = {2022},
  volume = {164},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v164/rudin22a.html},
}
```

### AMP (Adversarial Motion Priors)
```bibtex
@article{
  2021-TOG-AMP,
  author = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
  title = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
  journal = {ACM Trans. Graph.},
  issue_date = {August 2021},
  volume = {40},
  number = {4},
  month = jul,
  year = {2021},
  articleno = {1},
  numpages = {15},
  url = {http://doi.acm.org/10.1145/3450626.3459670},
  doi = {10.1145/3450626.3459670},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {motion control, physics-based character animation, reinforcement learning},
}
```

### Isaac Lab
请参考 [Isaac Lab 引用指南](https://isaaclab.github.io/citation.html)。

## 📄 许可证

本项目基于 BSD-3-Clause 许可证开源。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目基于以下优秀开源项目构建：

- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) - 高效的强化学习框架
- [Isaac Lab](https://isaaclab.github.io/) - 机器人仿真框架
- [Mujoco](https://mujoco.org/) - 物理引擎
- [AMP](https://github.com/xbpeng/amp) - 对抗运动先验

感谢所有贡献者和相关研究团队的工作！

---

**Legged Lab** - 让腿式机器人行走更自然、更智能！