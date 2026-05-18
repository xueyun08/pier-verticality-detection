"""
Phase 1 — 滑窗推理 + Logits 投票融合

将全场景推理改为与训练一致的 8m 窗口滑动推理,
每个点被多个窗口覆盖, 累积 softmax logits 后投票。
"""

import sys, os, time
import numpy as np
import torch
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from kpconv_model import KPConvUNet


def sliding_window_inference(model, points, device,
                              window_size=8.0, stride=2.0,
                              min_pts_per_window=100,
                              verbose=True):
    """
    对点云执行滑窗推理 + logits 投票。

    Parameters
    ----------
    model : KPConvUNet
    points : (N, 3) 全局点云坐标
    device : torch.device
    window_size : 窗口边长 (m), 应与训练 crop_radius 一致
    stride : 窗口步长 (m)
    min_pts_per_window : 窗口内最少点数
    verbose : 打印进度

    Returns
    -------
    pred_labels : (N,) 最终预测标签
    confidence  : (N,) 预测置信度
    """
    N = points.shape[0]
    half = window_size / 2
    n_classes = 2

    # 全局累积器
    global_logits = np.zeros((N, n_classes), dtype=np.float64)
    global_counts = np.zeros(N, dtype=np.int32)

    # 构建窗格网格
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    x_centers = np.arange(x_min + half, x_max, stride)
    y_centers = np.arange(y_min + half, y_max, stride)

    n_windows = len(x_centers) * len(y_centers)
    if verbose:
        print(f"滑窗推理: {len(x_centers)}x{len(y_centers)} = {n_windows} 窗口 "
              f"(window={window_size}m, stride={stride}m)")

    # 预建 KDTree 以加速窗口内点查询
    tree = cKDTree(points)

    win_idx = 0
    t0 = time.perf_counter()

    for cx in x_centers:
        for cy in y_centers:
            # 球形窗口 (与训练的球体 crop 保持一致)
            center = np.array([cx, cy, points[:, 2].mean()])

            # 用 KDTree 查窗口内点
            idx = np.array(tree.query_ball_point(center, r=half), dtype=np.int64)
            if len(idx) < min_pts_per_window:
                continue

            win_pts = points[idx].copy()

            # 去中心化 (与训练一致)
            win_pts -= center[None, :]

            # 限制点数 (与训练一致)
            max_pts = 5000
            if len(win_pts) > max_pts:
                keep = np.random.choice(len(win_pts), max_pts, replace=False)
                win_pts = win_pts[keep]
                idx = idx[keep]

            # 推理 (单窗口: 所有点属于同一个 batch)
            pts_tensor = torch.from_numpy(win_pts).float().to(device)
            batch_idx = torch.zeros(len(win_pts), dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(pts_tensor, batch_idx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (M, 2)

            # 全局累积
            global_logits[idx] += probs
            global_counts[idx] += 1

            win_idx += 1
            if verbose and win_idx % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {win_idx}/{n_windows} windows ({elapsed:.1f}s)")

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  完成: {win_idx} windows in {elapsed:.1f}s")

    # 投票: 取均值 logits
    valid = global_counts > 0
    final_logits = np.zeros((N, n_classes), dtype=np.float64)
    final_logits[valid] = global_logits[valid] / global_counts[valid, None]

    pred_labels = final_logits.argmax(axis=1).astype(np.int64)
    confidence = np.max(final_logits, axis=1)

    # 未被任何窗口覆盖的点 → 标为背景
    unseen = global_counts == 0
    if unseen.sum() > 0:
        if verbose:
            print(f"  {unseen.sum()} 点未被任何窗口覆盖, 标记为背景")

    return pred_labels, confidence, global_counts


def evaluate_scene(scene_name, npz_path, model, device, pier_class=8):
    """评估单个场景的全场景 IoU。"""
    print(f"\n{'='*60}")
    print(f"  场景: {scene_name}")
    print(f"{'='*60}")

    data = np.load(npz_path)
    raw_pts = data["points"].astype(np.float32)
    raw_lbl = data["labels"].ravel()

    # 体素降采样
    vsize = 0.3
    vidx = np.floor(raw_pts / vsize).astype(np.int64)
    _, inv_map, counts = np.unique(vidx, axis=0,
                                    return_inverse=True, return_counts=True)
    nv = inv_map.max() + 1
    sub_pts = np.zeros((nv, 3), dtype=np.float32)
    np.add.at(sub_pts, inv_map, raw_pts)
    sub_pts /= counts[:, None]

    # 标签转移
    tree = cKDTree(raw_pts)
    _, nn = tree.query(sub_pts, k=1)
    gt_labels = (raw_lbl[nn] == pier_class)

    print(f"  全局点: {nv:,} (voxel {vsize}m)")

    # 滑窗推理
    model.eval()
    pred, conf, coverage = sliding_window_inference(
        model, sub_pts, device, window_size=8.0, stride=2.0, verbose=True)

    # 指标
    pred_pier = (pred == 1)
    gt_pier = gt_labels

    inter = (pred_pier & gt_pier).sum()
    union = (pred_pier | gt_pier).sum()
    iou = inter / union if union > 0 else 0
    acc = (pred == gt_pier.astype(np.int64)).mean()

    tp = inter
    fp = pred_pier.sum() - inter
    fn = gt_pier.sum() - inter

    print(f"\n  GT pier:    {gt_pier.sum():,}")
    print(f"  Pred pier:  {pred_pier.sum():,}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  IoU:        {iou:.4f}")

    # 覆盖率统计
    covered = coverage > 0
    print(f"  覆盖: {covered.sum():,}/{nv:,} ({covered.sum()/nv*100:.1f}%)")
    print(f"  平均覆盖次数: {coverage[covered].mean():.1f}")

    return iou


# =========================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载 Phase 2 模型
    model = KPConvUNet(n_classes=2, n_kernel=15).to(device)
    ckpt = torch.load("./checkpoints/phase2_best.pth",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"模型: epoch={ckpt['epoch']}, block IoU={ckpt['iou']:.4f}\n")

    ious = {}
    for name in ["bridge_4", "bridge_6"]:
        ious[name] = evaluate_scene(
            name, f"data/{name}_parsed.npz", model, device)

    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    for name, iou in ious.items():
        print(f"  {name}: 全场景 IoU = {iou:.4f}")
