#!/usr/bin/env python
"""
train.py — KPConv Sim2Real 桥墩分割 两阶段训练主循环
============================================================

训练流程:
  Phase 1 (海量仿真预训练)
    - 使用 SyntheticBridgeDataset 在内存中源源不断地生成随机桥墩点云
    - 较大的学习率 (1e-2) 和较大的 Epoch 数, 让模型充分学习桥墩的几何多样性
    - 每个 Epoch 都包含全新的随机场景, 杜绝过拟合特定形状

  Phase 2 (真实域微调 — 占位)
    - 预留了读取真实 .pcd 文件的 DataLoader 接口
    - Phase 1 结束后自动将学习率降低 10 倍, 进入精细微调
    - 真实点云的加载逻辑用 TODO 标记, 后期放入 1~2 个手工标注的极恶劣真实点云

用法:
  # 仅跑 Phase 1 (仿真预训练)
  python train.py --phase1_epochs 50

  # 快速冒烟测试 (5 个 batch 不保存)
  python train.py --quick_test

  # 完整两阶段训练
  python train.py --phase1_epochs 100 --phase2_epochs 20 --real_data_dir ./real_pcd

依赖: torch, numpy, scipy, 以及项目内的 pier_scene_generator,
      degradation_engine, synthetic_bridge_dataset, kpconv_model
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 把 Algorithm 目录加入 path, 确保能 import 项目内的模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from synthetic_bridge_dataset import SyntheticBridgeDataset, collate_stacked
from kpconv_model import KPConvUNet, MixedLoss


# =========================================================================
#  辅助函数
# =========================================================================

def compute_metrics(logits, labels):
    """
    计算二分类语义分割指标。

    返回
    ----
    accuracy : float  整体像素准确率
    iou_pos  : float  前景 (桥墩, class=1) 的 IoU
    """
    preds = logits.argmax(dim=1)          # (N,)

    # 整体准确率
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.numel()

    # 前景 IoU
    pred_pos = (preds == 1)
    label_pos = (labels == 1)
    intersection = (pred_pos & label_pos).sum().float()
    union = (pred_pos | label_pos).sum().float()
    iou_pos = (intersection / (union + 1e-8)).item()

    return accuracy, iou_pos


def _phase_header(phase_name, total_epochs, lr):
    """打印 Phase 开始时美观的标题栏。"""
    print()
    print("=" * 64)
    print(f"  {phase_name}")
    print(f"  Epochs: {total_epochs}    Learning Rate: {lr:.1e}")
    print("=" * 64)


# =========================================================================
#  Phase 1 数据加载器
# =========================================================================

def build_synthetic_loader(batch_size, virtual_len, num_workers=0):
    """
    创建即时生成的合成数据 DataLoader。

    关键点：
    - Dataset 在 __getitem__ 中实时调用 PierSceneGenerator, 不读取任何磁盘文件
    - 每次访问都是全新的随机场景 → 无限数据流, 天然抗过拟合
    - collate_stacked 将变长点云拼接为 (total_N, 3) 的大矩阵,
      同时返回 batch_idx 标明每个点属于哪个场景
    """
    dataset = SyntheticBridgeDataset(virtual_len=virtual_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_stacked,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# =========================================================================
#  Phase 2 数据加载器 (真实点云 — 占位)
# =========================================================================

def build_real_loader(real_data_dir, batch_size):
    """
    真实 .pcd 点云 DataLoader —— 占位接口。

    后期在此补充完整的 .pcd 读取逻辑，预期格式:
      - 每个 .pcd 文件包含 labelled point cloud
      - 读取后转换为 (N, 3) points 和 (N,) labels 的 tensor
      - 使用与 Phase 1 相同的 collate_stacked 进行批处理

    TODO:
      1. 使用 open3d / pypcd 读取 .pcd 文件
      2. 将 label 字段映射为 {桥墩: 1, 背景: 0}
      3. 应用与训练时相同的归一化 / 增强策略
    """
    if real_data_dir is None or not os.path.isdir(real_data_dir):
        print(f"  [Phase 2] 真实数据目录 '{real_data_dir}' 不存在或为空 → 跳过 Phase 2")
        return None

    # ---- 占位: 这里后期替换为真正的 RealPointCloudDataset ----
    pcd_files = sorted(
        [f for f in os.listdir(real_data_dir) if f.endswith(".pcd")])

    if len(pcd_files) == 0:
        print(f"  [Phase 2] 目录 '{real_data_dir}' 中没有 .pcd 文件 → 跳过 Phase 2")
        return None

    print(f"  [Phase 2] 发现 {len(pcd_files)} 个真实 .pcd 文件: {pcd_files}")

    # 这里是占位 Dataset —— 后期替换为真实读取逻辑
    raise NotImplementedError(
        "真实 .pcd 读取逻辑尚未实现。请在 build_real_loader() 中完成 TODO 标记的部分。"
    )


# =========================================================================
#  单个 Epoch 的训练 / 验证
# =========================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=10):
    """
    在合成数据上跑一个 Epoch 的训练。

    由于数据是即时生成的, 没有 "验证集" 概念 ——
    每个 batch 都是全新场景, 所以训练的 loss / IoU 本身即反映泛化能力。
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    for batch_idx, (points, labels, batch_idx_tensor) in enumerate(loader):
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(points, batch_idx_tensor)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 统计
        acc, iou = compute_metrics(logits, labels)
        total_loss += loss.item()
        total_acc += acc
        total_iou += iou
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d} | "
                  f"Loss {total_loss/n_batches:.4f} | Acc {total_acc/n_batches:.4f} | "
                  f"IoU_pier {total_iou/n_batches:.4f} | {elapsed:.1f}s")

    return total_loss / n_batches, total_acc / n_batches, total_iou / n_batches


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """
    在有真实标注的验证集上评估 (Phase 2 微调时使用)。

    Phase 1 没有独立验证集, 此函数暂时搁置;
    当真实数据到位后即可启用。
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    n_batches = 0

    for points, labels, batch_idx_tensor in loader:
        points = points.to(device)
        labels = labels.to(device)
        logits = model(points, batch_idx_tensor)
        loss = criterion(logits, labels)

        acc, iou = compute_metrics(logits, labels)
        total_loss += loss.item()
        total_acc += acc
        total_iou += iou
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches, total_iou / n_batches


# =========================================================================
#  主训练入口
# =========================================================================

def main(args):
    # ---- 设备检测 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 实例化模型 ----
    model = KPConvUNet(n_classes=2, n_kernel=15).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # ---- 损失函数 ----
    # 使用 Focal + Dice 混合损失, 自动处理类别不平衡
    criterion = MixedLoss(alpha=0.25, gamma=2.0, dice_weight=0.5)

    # ---- 快速冒烟测试 ----
    if args.quick_test:
        print("\n  [快速冒烟测试] 仅跑 5 个 batch, 不保存模型。")
        # 小 batch + 空缓存, 适配小显存 GPU
        loader = build_synthetic_loader(batch_size=2, virtual_len=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i, (pts, lbl, bidx) in enumerate(loader):
            if i >= 5:
                break
            pts, lbl = pts.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model(pts, bidx)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            acc, iou = compute_metrics(logits, lbl)
            print(f"  Batch {i+1}: Loss={loss.item():.4f}  Acc={acc:.4f}  IoU={iou:.4f}")
        print("  冒烟测试通过。模型可正常前向/反向传播。")
        return

    # =====================================================================
    #  Phase 1: 海量仿真预训练
    # =====================================================================
    _phase_header("Phase 1 — 海量仿真预训练", args.phase1_epochs, args.phase1_lr)

    # 合成数据 DataLoader
    # virtual_len 设得很大 → 每Epoch有大量全新随机场景
    synth_loader = build_synthetic_loader(
        args.batch_size,
        virtual_len=args.virtual_len,
        num_workers=args.num_workers,
    )

    # 优化器 & 学习率调度
    optimizer = torch.optim.Adam(model.parameters(), lr=args.phase1_lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase1_epochs, eta_min=args.phase1_lr * 0.01)

    best_iou = 0.0
    phase1_start = time.perf_counter()

    for epoch in range(1, args.phase1_epochs + 1):
        print(f"\n--- Phase 1  Epoch {epoch}/{args.phase1_epochs}  "
              f"(lr={scheduler.get_last_lr()[0]:.2e}) ---")

        loss, acc, iou = train_one_epoch(
            model, synth_loader, criterion, optimizer, device, epoch,
            log_interval=args.log_interval)

        scheduler.step()

        print(f"  Epoch {epoch} 完成 | Loss {loss:.4f} | Acc {acc:.4f} | "
              f"IoU_pier {iou:.4f}")

        # 按 IoU 保存最佳模型
        if iou > best_iou and args.save_dir:
            best_iou = iou
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, "phase1_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iou": iou,
                "loss": loss,
            }, ckpt_path)
            print(f"  → 最佳模型已保存至 {ckpt_path} (IoU={iou:.4f})")

    phase1_elapsed = time.perf_counter() - phase1_start
    print(f"\nPhase 1 完成, 耗时 {phase1_elapsed/60:.1f} 分钟, "
          f"最佳 IoU={best_iou:.4f}")

    # =====================================================================
    #  Phase 2: 真实域微调 (占位 — 后期接入真实 .pcd 数据)
    # =====================================================================
    if args.phase2_epochs <= 0:
        print("\nPhase 2 跳过 (--phase2_epochs=0)。训练结束。")
        return

    _phase_header("Phase 2 — 真实域微调", args.phase2_epochs, args.phase2_lr)

    real_loader = build_real_loader(args.real_data_dir, args.batch_size)
    if real_loader is None:
        print("  真实数据不可用, 无法继续 Phase 2。训练结束。")
        return

    # 降低学习率 10 倍, 精细微调
    optimizer = torch.optim.Adam(model.parameters(), lr=args.phase2_lr,
                                  weight_decay=args.weight_decay * 0.5)

    best_iou_phase2 = 0.0

    for epoch in range(1, args.phase2_epochs + 1):
        print(f"\n--- Phase 2  Epoch {epoch}/{args.phase2_epochs}  "
              f"(lr={args.phase2_lr:.2e}) ---")

        loss, acc, iou = train_one_epoch(
            model, real_loader, criterion, optimizer, device, epoch,
            log_interval=args.log_interval)

        print(f"  Epoch {epoch} 完成 | Loss {loss:.4f} | Acc {acc:.4f} | "
              f"IoU_pier {iou:.4f}")

        if iou > best_iou_phase2 and args.save_dir:
            best_iou_phase2 = iou
            ckpt_path = os.path.join(args.save_dir, "phase2_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iou": iou,
                "loss": loss,
            }, ckpt_path)
            print(f"  → 最佳 Phase2 模型已保存至 {ckpt_path} (IoU={iou:.4f})")

    print(f"\nPhase 2 完成, 最佳 IoU={best_iou_phase2:.4f}")

    # ---- 最终模型保存 ----
    if args.save_dir:
        final_path = os.path.join(args.save_dir, "final_model.pth")
        torch.save({"model_state_dict": model.state_dict()}, final_path)
        print(f"最终模型已保存至 {final_path}")

    print("\n训练全部完成。")


# =========================================================================
#  CLI 参数
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="KPConv Sim2Real 桥墩分割 — 两阶段训练")

    # ---- Phase 1 ----
    p.add_argument("--phase1_epochs", type=int, default=100,
                   help="Phase 1 仿真预训练 Epoch 数")
    p.add_argument("--phase1_lr", type=float, default=1e-2,
                   help="Phase 1 初始学习率 (默认 1e-2)")

    # ---- Phase 2 ----
    p.add_argument("--phase2_epochs", type=int, default=20,
                   help="Phase 2 真实域微调 Epoch 数 (0 = 跳过)")
    p.add_argument("--phase2_lr", type=float, default=1e-3,
                   help="Phase 2 学习率 (默认 1e-3, 为 Phase1 的 1/10)")
    p.add_argument("--real_data_dir", type=str, default=None,
                   help="真实 .pcd 点云目录 (Phase 2 使用)")

    # ---- 数据 ----
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch 大小 (默认 4)")
    p.add_argument("--virtual_len", type=int, default=500,
                   help="合成数据一个 Epoch 内遍历的虚拟样本数 (默认 500)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader 工作进程 (Windows 建议 0, Linux 建议 4)")

    # ---- 优化器 ----
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Adam 权重衰减")

    # ---- 日志与存储 ----
    p.add_argument("--log_interval", type=int, default=10,
                   help="每隔 N 个 batch 打印一次日志")
    p.add_argument("--save_dir", type=str, default="./checkpoints",
                   help="模型保存目录")

    # ---- 调试 ----
    p.add_argument("--quick_test", action="store_true",
                   help="快速冒烟测试: 跑 5 个 batch 后退出")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
