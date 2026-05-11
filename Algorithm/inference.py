"""
inference.py — 使用预训练 KPConv 模型对真实点云进行桥墩分割推理

支持的格式: .pcd, .las, .laz, .ply, .xyz, .txt, .npy
输出: 彩色点云 (.ply) + 统计报告

用法:
  python inference.py --input path/to/cloud.pcd
  python inference.py --input path/to/cloud.las --voxel_size 0.3
  python inference.py --input path/to/cloud.ply --output result.ply
"""

import os
import sys
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from kpconv_model import KPConvUNet


# =========================================================================
#  点云加载
# =========================================================================

def load_point_cloud(filepath):
    """
    自动检测格式并加载点云, 返回 (N, 3) float32 numpy 数组。

    支持: .pcd / .las / .laz / .ply / .xyz / .txt / .npy
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".las", ".laz"):
        return _load_las(filepath)

    elif ext == ".pcd":
        return _load_pcd(filepath)

    elif ext == ".ply":
        return _load_ply(filepath)

    elif ext in (".xyz", ".txt"):
        return _load_xyz(filepath)

    elif ext == ".npy":
        arr = np.load(filepath)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
        raise ValueError(f".npy 文件 shape={arr.shape}, 需要 (N, ≥3)")

    else:
        # 回退: 尝试当文本文件加载
        return _load_xyz(filepath)


def _load_las(filepath):
    import laspy
    las = laspy.read(filepath)
    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    return np.column_stack([x, y, z])


def _load_pcd(filepath):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points, dtype=np.float32)


def _load_ply(filepath):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points, dtype=np.float32)


def _load_xyz(filepath):
    pts = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)


# =========================================================================
#  点云预处理
# =========================================================================

def preprocess(points, voxel_size=0.2, center=True):
    """
    预处理真实点云:
      1. 去中心化 (可选 — 模型训练时坐标以桥墩为中心)
      2. 体素降采样 (控制点数, 加速推理)

    返回
    ----
    pts       : (M, 3)  降采样后的点云
    offset    : (3,)    去中心化偏移量 (还原时加回去)
    voxel_map : None    预留, 后期可用于点云还原
    """
    if center:
        offset = points.mean(axis=0)
        offset[2] = points[:, 2].min()  # Z 保持地面为基准
        pts = points - offset[None, :]
    else:
        offset = np.zeros(3, dtype=np.float32)

    if voxel_size is not None and voxel_size > 0:
        voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
        _, inv_map = np.unique(voxel_idx, axis=0, return_inverse=True)
        n_voxels = inv_map.max() + 1
        sub_pts = np.zeros((n_voxels, 3), dtype=np.float32)
        np.add.at(sub_pts, inv_map, pts)
        counts = np.bincount(inv_map, minlength=n_voxels).astype(np.float32)
        pts = sub_pts / counts[:, None]

    return pts.astype(np.float32), offset


# =========================================================================
#  模型推理
# =========================================================================

@torch.no_grad()
def predict(model, points, device, batch_size=6000):
    """
    对点云运行 KPConv 分割推理。

    对于大数据 (> batch_size 点), 采用空间分块策略逐块推理,
    避免 cKDTree 在超大规模点集上的 O(N²) 开销。

    返回
    ----
    labels : (N,)  int64  — 0=背景, 1=桥墩
    logits : (N, 2) float32
    """
    N = points.shape[0]

    # ---- 直接推理 (小点云) ----
    if N <= batch_size:
        pts_t = torch.from_numpy(points).to(device)
        batch_idx = torch.zeros(N, dtype=torch.long)
        logits = model(pts_t, batch_idx)
        preds = logits.argmax(dim=1).cpu().numpy()
        return preds, logits.cpu().numpy()

    # ---- 空间分块推理 (大点云) ----
    print(f"  点云较大 ({N} 点), 使用空间分块 (块大小={batch_size}) ...")
    labels = np.zeros(N, dtype=np.int64)
    logits_all = np.zeros((N, 2), dtype=np.float32)

    # 用 XYZ 坐标排序, 相邻块有空间连续性
    order = np.lexsort((points[:, 0], points[:, 1], points[:, 2]))
    pts_sorted = points[order]

    n_chunks = int(np.ceil(N / batch_size))
    for i in range(n_chunks):
        start = i * batch_size
        end = min(start + batch_size, N)
        chunk = pts_sorted[start:end]
        pts_t = torch.from_numpy(chunk).to(device)
        bidx = torch.zeros(chunk.shape[0], dtype=torch.long)
        logits = model(pts_t, bidx)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels[order[start:end]] = preds
        logits_all[order[start:end]] = logits.cpu().numpy()

        if (i + 1) % 10 == 0:
            print(f"    chunk {i+1}/{n_chunks} ...")

    return labels, logits_all


# =========================================================================
#  结果输出
# =========================================================================

def save_colored_ply(filepath, points, labels):
    """
    保存着色 .ply 文件: 桥墩=红色, 背景=灰色。
    可用 CloudCompare / MeshLab / open3d 打开查看。
    """
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors[labels == 1] = [220, 40, 40]     # 桥墩: 红色
    colors[labels == 0] = [128, 128, 128]   # 背景: 灰色

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"\n已保存着色点云: {filepath}")


def print_stats(labels):
    """打印分割统计。"""
    total = len(labels)
    n_pier = (labels == 1).sum()
    n_bg = (labels == 0).sum()
    print(f"\n{'='*50}")
    print(f"  推理结果统计")
    print(f"{'='*50}")
    print(f"  总点数    : {total:,}")
    print(f"  桥墩 (1)  : {n_pier:,}  ({n_pier/total*100:.1f}%)")
    print(f"  背景 (0)  : {n_bg:,}  ({n_bg/total*100:.1f}%)")
    print(f"{'='*50}")


# =========================================================================
#  主入口
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="使用预训练 KPConv 模型对真实点云进行桥墩分割推理")
    parser.add_argument("--input", "-i", required=True,
                        help="输入点云路径 (.pcd/.las/.ply/.xyz/.txt/.npy)")
    parser.add_argument("--output", "-o", default=None,
                        help="输出着色 .ply 文件路径 (默认: <input>_pred.ply)")
    parser.add_argument("--checkpoint", "-c",
                        default="./checkpoints/phase1_best.pth",
                        help="预训练权重路径")
    parser.add_argument("--voxel_size", "-v", type=float, default=0.2,
                        help="预处理降采样体素大小 (m), 0=不降采样")
    parser.add_argument("--chunk_size", type=int, default=6000,
                        help="推理时每个 chunk 的最大点数")
    parser.add_argument("--no_center", action="store_true",
                        help="不去中心化 (点云已经以目标物为中心)")
    args = parser.parse_args()

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 加载模型 ----
    print(f"加载模型: {args.checkpoint}")
    model = KPConvUNet(n_classes=2, n_kernel=15).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  模型 IoU (训练时): {ckpt.get('iou', 'N/A')}")

    # ---- 加载点云 ----
    print(f"\n加载点云: {args.input}")
    raw_pts = load_point_cloud(args.input)
    print(f"  原始点数: {raw_pts.shape[0]:,}")
    print(f"  X: [{raw_pts[:,0].min():.1f}, {raw_pts[:,0].max():.1f}]")
    print(f"  Y: [{raw_pts[:,1].min():.1f}, {raw_pts[:,1].max():.1f}]")
    print(f"  Z: [{raw_pts[:,2].min():.1f}, {raw_pts[:,2].max():.1f}]")

    # ---- 预处理 ----
    pts, offset = preprocess(raw_pts, voxel_size=args.voxel_size,
                             center=not args.no_center)
    print(f"\n预处理后: {pts.shape[0]:,} 点 (voxel={args.voxel_size}m)")

    # ---- 推理 ----
    print("\n开始推理 ...")
    labels, _logits = predict(model, pts, device, batch_size=args.chunk_size)

    # ---- 输出 ----
    print_stats(labels)

    output_path = args.output or os.path.splitext(args.input)[0] + "_pred.ply"
    save_colored_ply(output_path, pts, labels)

    # ---- 显示前 5 个预测为桥墩的点坐标 ----
    pier_pts = pts[labels == 1]
    if len(pier_pts) > 0:
        print(f"\n桥墩点质心: [{pier_pts[:,0].mean():.2f}, "
              f"{pier_pts[:,1].mean():.2f}, {pier_pts[:,2].mean():.2f}]")
        print(" (此为去中心化后的坐标, 加 offset 可还原真实坐标)")
    else:
        print("\n  ⚠ 未检测到桥墩点! 可能原因:")
        print("    1. 场景中确实没有桥墩")
        print("    2. 点云跨度过大, 桥墩尺寸与训练数据不匹配")
        print("    3. 尝试调小 --voxel_size 保留更多细节")


if __name__ == "__main__":
    main()
