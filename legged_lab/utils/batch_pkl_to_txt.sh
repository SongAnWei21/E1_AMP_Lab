#!/bin/bash

# ==============================================================================
# 配置区域：请根据你的实际情况修改下面两个路径
# ==============================================================================
# 1. 存放所有 .pkl 文件的源文件夹
INPUT_DIR="/home/saw/droidup/TienKung-Lab/legged_lab/envs/e1_19dof/datasets/walk_30fps"

# 2. 生成的 .txt 文件要保存的目标文件夹
# 建议保存在你的强化学习环境目录下，例如：
OUTPUT_DIR="/home/saw/droidup/TienKung-Lab/legged_lab/envs/e1_19dof/datasets/motion_visualization"
# ==============================================================================

# 确保输出文件夹存在
mkdir -p "$OUTPUT_DIR"

echo "🚀 开始批量转换 PKL 到 TXT..."
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=================================================="

# 计数器
count=0

# 遍历输入文件夹下的所有 .pkl 文件
for pkl_file in "$INPUT_DIR"/*.pkl; do
    # 检查是否存在文件（防止文件夹为空时报错）
    if [ ! -e "$pkl_file" ]; then
        echo "❌ 在输入目录中没有找到 .pkl 文件！"
        break
    fi

    # 提取纯文件名（不带路径）和去掉后缀的名字
    filename=$(basename -- "$pkl_file")
    name_no_ext="${filename%.*}"

    # 拼接最终的 .txt 输出路径
    output_txt="$OUTPUT_DIR/${name_no_ext}.txt"

    echo "⏳ 正在转换: $filename -> ${name_no_ext}.txt"
    
    # 执行 Python 转换脚本
    # 假设你是在 TienKung-Lab 项目根目录下运行这个 bash 脚本
    python legged_lab/scripts/gmr_data_conversion.py \
        --input_pkl "$pkl_file" \
        --output_txt "$output_txt"

    # 检查上一条命令是否执行成功
    if [ $? -eq 0 ]; then
        echo "✅ 成功保存至: $output_txt"
        ((count++))
    else
        echo "❌ 转换失败: $filename"
    fi
    echo "--------------------------------------------------"
done

echo "🎉 批量转换完成！共成功处理 $count 个文件。"