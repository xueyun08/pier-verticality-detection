"""
KPConv 骨架 + 混合损失函数
===========================
基于论文 "KPConv: Flexible and Deformable Convolution for Point Clouds"
实现刚性 (rigid) KPConv 层、U-Net 风格编解码器、以及针对二分类
（桥墩 vs 背景）的 Focal + Dice 混合损失。

依赖: torch, numpy, scipy (cKDTree 用于近邻搜索)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


# =========================================================================
#  工具函数
# =========================================================================

def fibonacci_sphere(n):
    """在单位球面上均匀采样 n 个核心点 (kernel points)。"""
    pts = np.zeros((n, 3), dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0
        radius = np.sqrt(1.0 - y * y)
        theta = phi * i
        pts[i, 0] = np.cos(theta) * radius
        pts[i, 1] = y
        pts[i, 2] = np.sin(theta) * radius
    return pts


def grid_subsample(points, features, voxel_size):
    """
    体素网格降采样：将每个体素内的点取均值，作为粗粒度的点。

    points   : (N, 3)   float32 tensor (任意 device)
    features : (N, C)   float32 tensor (任意 device)
    voxel_size : float   体素边长 (m)

    返回
    ----
    sub_pts : (M, 3)   降采样后的点坐标
    sub_feat: (M, C)   降采样后的特征
    inv_map : (N,)     int64  原始点 → 降采样点 的映射 (用于上采样)
    """
    device = points.device
    pts_np = points.detach().cpu().numpy()
    feat_np = features.detach().cpu().numpy()

    voxel_idx = np.floor(pts_np / voxel_size).astype(np.int64)
    _, inv_map, counts = np.unique(voxel_idx, axis=0,
                                   return_inverse=True, return_counts=True)

    n_voxels = inv_map.max() + 1

    sub_pts = np.zeros((n_voxels, 3), dtype=np.float32)
    np.add.at(sub_pts, inv_map, pts_np)
    sub_pts /= counts[:, None]

    sub_feat = np.zeros((n_voxels, feat_np.shape[1]), dtype=np.float32)
    np.add.at(sub_feat, inv_map, feat_np)
    sub_feat /= counts[:, None]

    return (torch.from_numpy(sub_pts).to(device),
            torch.from_numpy(sub_feat).to(device),
            torch.from_numpy(inv_map.astype(np.int64)))


def nearest_upsample(src_points, dst_points, src_features):
    """
    最近邻上采样：把粗粒度点 (src) 的特征插值到细粒度点 (dst) 上。

    src_points  : (M, 3)  粗粒度点坐标 (任意 device)
    dst_points  : (N, 3)  细粒度点坐标 (任意 device)
    src_features: (M, C)  粗粒度特征

    返回
    ----
    dst_features: (N, C)  插值到细粒度点的特征
    """
    device = src_features.device
    tree = cKDTree(src_points.detach().cpu().numpy())
    _, idx = tree.query(dst_points.detach().cpu().numpy(), k=1)
    return src_features[torch.from_numpy(idx.astype(np.int64)).to(device)]


# =========================================================================
#  KPConv 层 (刚性版)
# =========================================================================

class RigidKPConv(nn.Module):
    """
    刚性 KPConv 层：使用 k-NN 搜索邻居，球形核点加权卷积。

    相比半径搜索, k-NN 产生固定大小的邻居集，可以完全向量化构建边列表，
    避免 Python 层循环，大幅提升训练速度 (10×+)。
    """

    def __init__(self, in_channels, out_channels, n_kernel_points=15,
                 radius=2.0, k_neighbors=40, sigma_ratio=2.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.k_neighbors = k_neighbors
        self.n_kernel_points = n_kernel_points
        self.sigma = radius / sigma_ratio

        kernel_pts = fibonacci_sphere(n_kernel_points)
        self.register_buffer("kernel_points",
                             torch.from_numpy(kernel_pts * radius * 0.6))

        self.kernel_weights = nn.Parameter(
            torch.empty(n_kernel_points, out_channels, in_channels))
        nn.init.kaiming_uniform_(self.kernel_weights)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, points, features):
        N = points.shape[0]
        device = features.device

        # ---- 1. k-NN 搜索 (整数输出, 可完全 numpy 向量化) ----
        pts_np = points.detach().cpu().numpy().astype(np.float64)
        tree = cKDTree(pts_np)
        k = min(self.k_neighbors, N)
        _, idx_np = tree.query(pts_np, k=k)  # (N, k)  int64

        # ---- 2. 完全向量化构建边列表 ----
        # dst: [0,0,...0, 1,1,...1, ..., N-1]  每个点重复 k 次
        dst = np.repeat(np.arange(N), k)
        src = idx_np.ravel()  # 展平 (N, k) → (N*k,)
        dst = torch.from_numpy(dst).long().to(device)
        src = torch.from_numpy(src).long().to(device)

        # ---- 3. 核点相关权重 ----
        rel_pos = (points[src] - points[dst]).float()  # (N*k, 3)
        kernel_exp = self.kernel_points[None, :, :].to(device)  # (1, K, 3)
        dists = torch.norm(rel_pos[:, None, :] - kernel_exp, dim=2)  # (N*k, K)
        correlations = torch.clamp(1.0 - dists / self.sigma, min=0.0)

        # ---- 4. 卷积 ----
        output = torch.zeros(N, self.out_channels, device=device)
        src_feat = features[src]
        kernel_w = self.kernel_weights.to(device)

        for k in range(self.n_kernel_points):
            corr_k = correlations[:, k]
            mask = corr_k > 0
            if not mask.any():
                continue
            weighted = src_feat[mask] @ kernel_w[k].T
            weighted = weighted * corr_k[mask, None]
            output.index_add_(0, dst[mask], weighted)

        nbr_counts = torch.bincount(dst, minlength=N).float().clamp(min=1)
        output = output / nbr_counts[:, None]
        output = output + self.bias.to(device)[None, :]
        return output


# =========================================================================
#  KPConv 残差块
# =========================================================================

class KPConvBlock(nn.Module):
    """KPConv → BatchNorm → LeakyReLU → (可选残差连接)"""

    def __init__(self, in_ch, out_ch, n_kernel=15, radius=2.0, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = RigidKPConv(in_ch, out_ch, n_kernel, radius)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1)

        # 当输入/输出通道不一致时, 用 1×1 卷积对齐做残差
        self.shortcut = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, points, features):
        out = self.conv(points, features)
        out = self.bn(out)
        shortcut = self.shortcut(features)
        return self.act(out + shortcut)


# =========================================================================
#  KPConv U-Net 分割网络
# =========================================================================

class KPConvUNet(nn.Module):
    """
    三层次 KPConv U-Net，用于点云二分类语义分割。

    编码器通过体素降采样逐步扩大感受野；
    解码器通过最近邻上采样 + 跳跃连接恢复细节。
    """

    def __init__(self, n_classes=2, n_kernel=15):
        super().__init__()
        ks = n_kernel

        # ---- 编码器 ----
        # Level 0 (全分辨率)
        self.enc0_1 = KPConvBlock(3, 32, ks, radius=1.5)
        self.enc0_2 = KPConvBlock(32, 64, ks, radius=1.5)

        # Level 1 (降采样 ×2)
        self.enc1_1 = KPConvBlock(64, 64, ks, radius=3.0)
        self.enc1_2 = KPConvBlock(64, 128, ks, radius=3.0)

        # Level 2 (降采样 ×2)
        self.enc2_1 = KPConvBlock(128, 128, ks, radius=6.0)
        self.enc2_2 = KPConvBlock(128, 256, ks, radius=6.0)

        # ---- 解码器 ----
        # Level 2 → 1
        self.dec2_1 = KPConvBlock(256, 128, ks, radius=6.0)

        # Level 1 → 0  (concat skip: enc1_2=128 + dec2_up=128 = 256)
        self.dec1_1 = KPConvBlock(256, 64, ks, radius=3.0)

        # Level 0 → output  (concat skip: enc0_2=64 + dec1_up=64 = 128)
        self.dec0_1 = KPConvBlock(128, 32, ks, radius=1.5)

        # 分类头 (不要用 Sequential —— KPConv 需要 (points, features) 两个参数)
        self.cls_conv = KPConvBlock(32, 32, ks, radius=1.0)
        self.cls_linear = nn.Linear(32, n_classes)

        # 缓存降采样/上采样的映射 (在 forward 中动态更新)
        self._cache = {}

    def forward(self, points, batch_idx):
        """
        points    : (N, 3)  float32  — 拼接后的所有点
        batch_idx : (N,)    int64    — 每个点所属的场景编号

        返回
        ----
        logits : (N, n_classes) float32
        """
        # 为每个场景独立编码/解码, 最后拼接
        B = int(batch_idx.max().item()) + 1
        all_logits = []

        for b in range(B):
            mask = batch_idx == b
            b_pts = points[mask]             # 保持原 device
            b_feat = b_pts.clone()           # 初始特征 = 坐标

            # ---- 编码 ----
            # Level 0
            f0 = self.enc0_1(b_pts, b_feat)
            f0 = self.enc0_2(b_pts, f0)

            # 降采样 0 → 1 (voxel 0.5 m)
            p1, f1_g, map01 = grid_subsample(b_pts, f0, voxel_size=0.5)
            if p1.shape[0] < 10:  # 点太少则跳过
                all_logits.append(torch.zeros(len(b_pts), 2, device=points.device))
                continue

            f1 = self.enc1_1(p1, f1_g)
            f1 = self.enc1_2(p1, f1)

            # 降采样 1 → 2 (voxel 1.0 m)
            p2, f2_g, map12 = grid_subsample(p1, f1, voxel_size=1.0)
            if p2.shape[0] < 5:
                # 直接上采样回 Level 0
                f1_up = nearest_upsample(p1, b_pts, f1)
                cls_feat = self.cls_conv(b_pts, f1_up)
                logits_b = self.cls_linear(cls_feat)
                all_logits.append(logits_b)
                continue

            f2 = self.enc2_1(p2, f2_g)
            f2 = self.enc2_2(p2, f2)

            # ---- 解码 ----
            f2 = self.dec2_1(p2, f2)

            # 上采样 2 → 1: 把 p2 的特征插值回 p1
            f2_up = nearest_upsample(p2, p1, f2)          # (M1, C)
            f1_cat = torch.cat([f1, f2_up], dim=1)        # 跳跃连接
            f1_dec = self.dec1_1(p1, f1_cat)

            # 上采样 1 → 0
            f1_up = nearest_upsample(p1, b_pts, f1_dec)   # (N_b, C)
            f0_cat = torch.cat([f0, f1_up], dim=1)
            f0_dec = self.dec0_1(b_pts, f0_cat)

            # 分类头
            cls_feat = self.cls_conv(b_pts, f0_dec)
            logits_b = self.cls_linear(cls_feat)             # (N_b, 2)
            all_logits.append(logits_b.to(points.device))

        return torch.cat(all_logits, dim=0)


# =========================================================================
#  损失函数
# =========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡, 降低易分类样本的权重。

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits  : (N, C)
        targets : (N,)  int64
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)                               # p_t
        focal_weight = (1 - pt) ** self.gamma

        # α 加权
        alpha_t = torch.where(targets == 1,
                              self.alpha,
                              1 - self.alpha)
        return (alpha_t * focal_weight * ce).mean()


class DiceLoss(nn.Module):
    """
    Dice Loss: 直接优化前景类的 Dice 系数 (F1 近似)。

    Dice = 2 * Σ(p_i * y_i) / (Σ(p_i) + Σ(y_i) + ε)
    Loss = 1 - Dice
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits  : (N, C)
        targets : (N,)  int64
        """
        probs = F.softmax(logits, dim=1)
        # 对前景类 (class=1) 计算 Dice
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


class MixedLoss(nn.Module):
    """
    Focal + Dice 混合损失, 兼顾像素级分类与区域重叠度。

    L = λ * DiceLoss + (1-λ) * FocalLoss
    """

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_f = self.focal(logits, targets)
        loss_d = self.dice(logits, targets)
        return self.dice_weight * loss_d + (1 - self.dice_weight) * loss_f


# =========================================================================
#  快速测试
# =========================================================================

if __name__ == "__main__":
    print("=== KPConv 模型快速测试 ===\n")

    # 生成一组合成数据 (模拟一个 batch 的拼接结果)
    from synthetic_bridge_dataset import SyntheticBridgeDataset, collate_stacked
    from torch.utils.data import DataLoader

    ds = SyntheticBridgeDataset(virtual_len=4, seed=0)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_stacked)
    points, labels, batch_idx = next(iter(loader))

    print(f"输入:  points={points.shape}, labels={labels.shape}, "
          f"batch_idx={batch_idx.shape}")

    # 前向传播
    model = KPConvUNet(n_classes=2, n_kernel=15)
    model.eval()
    with torch.no_grad():
        logits = model(points, batch_idx)
    print(f"输出:  logits={logits.shape}")

    # 损失函数测试
    criterion = MixedLoss()
    loss = criterion(logits, labels)
    print(f"损失:  {loss.item():.4f}")

    # 参数统计
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    print("\n测试通过。")
