"""
Phase 2 — 真实域微调

在 bridge_4 真实点云上微调 Phase 1 预训练模型。
- 桥墩: Class 8
- 学习率: 1e-3 (Phase 1 的 1/10)
- 空间块采样, 每块 30m × 30m
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from real_bridge_dataset import RealBridgeDataset
from synthetic_bridge_dataset import collate_stacked
from kpconv_model import KPConvUNet, MixedLoss


def compute_metrics(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.numel()
    pred_pos = (preds == 1)
    label_pos = (labels == 1)
    intersection = (pred_pos & label_pos).sum().float()
    union = (pred_pos | label_pos).sum().float()
    iou_pos = (intersection / (union + 1e-8)).item()
    return accuracy, iou_pos


def train_phase2(model, loader, criterion, optimizer, device, epoch):
    """在真实数据上跑一个 epoch。"""
    model.train()
    metrics = defaultdict(float)
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

        acc, iou = compute_metrics(logits, labels)
        metrics["loss"] += loss.item()
        metrics["acc"] += acc
        metrics["iou"] += iou
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d} | "
                  f"Loss {metrics['loss']/n_batches:.4f} | "
                  f"Acc {metrics['acc']/n_batches:.4f} | "
                  f"IoU {metrics['iou']/n_batches:.4f} | "
                  f"{elapsed:.1f}s")

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 2 真实域微调")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/phase1_best.pth")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"Phase 2 — 真实域微调 | LR={args.lr:.1e} | Epochs={args.epochs}")

    # ---- 数据 ----
    npz_paths = ["data/bridge_4_parsed.npz", "data/bridge_6_parsed.npz"]
    missing = [p for p in npz_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing: {missing}")
        return

    dataset = RealBridgeDataset(npz_paths, pier_class=8, virtual_len=600)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=collate_stacked, num_workers=0)

    # ---- 模型 ----
    model = KPConvUNet(n_classes=2, n_kernel=15).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"加载预训练权重: {args.checkpoint} (Phase1 IoU={ckpt['iou']:.4f})")

    # ---- 优化器 ----
    criterion = MixedLoss(alpha=0.25, gamma=2.0, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # ---- 微调 ----
    best_iou = 0.0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Phase 2  Epoch {epoch}/{args.epochs}  "
              f"(lr={scheduler.get_last_lr()[0]:.2e}) ---")

        metrics = train_phase2(model, loader, criterion, optimizer, device, epoch)
        scheduler.step()

        print(f"  Epoch {epoch} 完成 | Loss {metrics['loss']:.4f} | "
              f"Acc {metrics['acc']:.4f} | IoU {metrics['iou']:.4f}")

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "phase2_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "iou": best_iou,
                "loss": metrics["loss"],
            }, save_path)
            print(f"  -> 保存最佳模型: {save_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nPhase 2 完成, 耗时 {elapsed/60:.1f} 分钟, 最佳 IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
