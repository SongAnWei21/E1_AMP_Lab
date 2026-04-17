import pickle
import argparse
import numpy as np

def crop_pkl(input_path, output_path, start_frame, end_frame):
    print(f"正在加载文件: {input_path} ...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    cropped_data = {}
    
    print(f"\n正在裁剪帧区间: [{start_frame} : {end_frame}]")
    print("-" * 50)
    
    # 自动遍历字典，对所有 Numpy 数组沿第一维（时间维度 T）进行裁剪
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            # 切片操作
            cropped_data[key] = value[start_frame:end_frame]
            print(f"{key:<15} | 形状: {value.shape} -> {cropped_data[key].shape}")
        else:
            # 标量、字符串、字典等元数据直接保留 (比如 fps, meta, 名字等)
            cropped_data[key] = value
            
    # （可选）更新一下记录的文件名，方便追溯
    if "motion_file" in cropped_data and isinstance(cropped_data["motion_file"], str):
        cropped_data["motion_file"] = f"{cropped_data['motion_file']}_crop_{start_frame}_{end_frame}"

    # 保存新的 pkl
    print("-" * 50)
    print(f"正在保存至: {output_path} ...")
    with open(output_path, 'wb') as f:
        pickle.dump(cropped_data, f)
        
    print("✅ 裁剪完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裁剪 GMR 导出的 .pkl 动捕文件帧数")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 .pkl 文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 .pkl 文件路径")
    parser.add_argument("--start", "-s", type=int, required=True, help="起始帧 (包含)")
    parser.add_argument("--end", "-e", type=int, required=True, help="结束帧 (不包含)")
    
    args = parser.parse_args()
    
    # 简单的逻辑防呆
    if args.start >= args.end:
        raise ValueError(f"起始帧 ({args.start}) 必须小于结束帧 ({args.end})！")
        
    crop_pkl(args.input, args.output, args.start, args.end)