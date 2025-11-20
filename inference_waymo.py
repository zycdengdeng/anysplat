#!/usr/bin/env python3
"""
AnySplat 推理脚本 - 适配 Waymo 数据集
支持两种模式：
1. 多视角模式：使用同一时刻的多个相机视角
2. 时序模式：使用同一相机的多个时间步
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from glob import glob

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image


def load_waymo_images_multiview(image_dir, timestamp, cameras=[0, 1, 2, 3, 4]):
    """
    加载同一时刻的多个相机视角

    Args:
        image_dir: 图片目录
        timestamp: 时间戳，如 "000000"
        cameras: 相机列表，默认 [0, 1, 2, 3, 4]
    """
    image_paths = []
    for cam_id in cameras:
        img_path = os.path.join(image_dir, f"{timestamp}_{cam_id}.png")
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            print(f"警告: 图片不存在 {img_path}")

    return image_paths


def load_waymo_images_temporal(image_dir, camera_id, start_frame, num_frames):
    """
    加载同一相机的多个时间步

    Args:
        image_dir: 图片目录
        camera_id: 相机ID，0-4
        start_frame: 起始帧号
        num_frames: 帧数
    """
    image_paths = []
    for i in range(num_frames):
        frame_id = start_frame + i
        timestamp = f"{frame_id:06d}"
        img_path = os.path.join(image_dir, f"{timestamp}_{camera_id}.png")
        if os.path.exists(img_path):
            image_paths.append(img_path)
        else:
            print(f"警告: 图片不存在 {img_path}")

    return image_paths


def main():
    parser = argparse.ArgumentParser(description='AnySplat Waymo 数据推理')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Waymo 图片目录路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--mode', type=str, default='multiview',
                        choices=['multiview', 'temporal'],
                        help='推理模式: multiview(多视角) 或 temporal(时序)')

    # 多视角模式参数
    parser.add_argument('--timestamp', type=str, default='000000',
                        help='时间戳 (多视角模式)')
    parser.add_argument('--cameras', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='相机ID列表 (多视角模式)')

    # 时序模式参数
    parser.add_argument('--camera_id', type=int, default=0,
                        help='相机ID (时序模式)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='起始帧号 (时序模式)')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='帧数 (时序模式)')

    # 模型参数
    parser.add_argument('--gpu', type=int, default=6,
                        help='使用的GPU ID')
    parser.add_argument('--model_name', type=str, default='lhjiang/anysplat',
                        help='Hugging Face 模型名称')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 根据模式加载图片
    if args.mode == 'multiview':
        print(f"=== 多视角模式 ===")
        print(f"时间戳: {args.timestamp}")
        print(f"相机: {args.cameras}")
        image_paths = load_waymo_images_multiview(
            args.image_dir, args.timestamp, args.cameras
        )
        output_name = f"waymo_multiview_{args.timestamp}"
    else:  # temporal
        print(f"=== 时序模式 ===")
        print(f"相机ID: {args.camera_id}")
        print(f"帧范围: {args.start_frame} - {args.start_frame + args.num_frames - 1}")
        image_paths = load_waymo_images_temporal(
            args.image_dir, args.camera_id, args.start_frame, args.num_frames
        )
        output_name = f"waymo_temporal_cam{args.camera_id}_f{args.start_frame}"

    if len(image_paths) == 0:
        print("错误: 没有找到任何图片！")
        return

    print(f"\n找到 {len(image_paths)} 张图片:")
    for i, path in enumerate(image_paths):
        print(f"  [{i}] {os.path.basename(path)}")

    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 加载模型
    print(f"\n正在加载模型 {args.model_name}...")
    model = AnySplat.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("✓ 模型加载成功")

    # 预处理图片
    print("\n正在预处理图片...")
    images = []
    for img_path in image_paths:
        img = process_image(img_path)
        images.append(img)

    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape
    print(f"图片张量形状: {images.shape}")

    # 运行推理
    print("\n正在运行推理...")
    with torch.no_grad():
        gaussians, pred_context_pose = model.inference((images + 1) * 0.5)

    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']
    print("✓ 推理完成")

    # 保存结果
    print(f"\n正在保存结果到 {args.output_dir}...")
    output_path = os.path.join(args.output_dir, output_name)
    save_interpolated_video(
        pred_all_extrinsic,
        pred_all_intrinsic,
        b, h, w,
        gaussians,
        output_path,
        model.decoder
    )
    print(f"✓ 结果已保存到: {output_path}")

    print("\n=== 推理完成！ ===")


if __name__ == "__main__":
    main()
