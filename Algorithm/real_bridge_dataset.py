"""
Phase 2 真实点云数据集 — 从标注好的大场景中提取训练样本。

将 8.9M 点的大场景切分为空间块 (spatial crops)，
每个块独立送入 KPConv 进行微调。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RealBridgeDataset(Dataset):
    """
    真实桥墩点云数据集, 从单一大场景中随机空间切块。

    Parameters
    ----------
    parsed_npz : str
        bridge_4_parsed.npz 文件路径
    pier_class : int
        桥墩对应的标签值 (Class 8)
    crop_radius : float
        空间块的半边长 (m), 默认 15 → 30m × 30m 块
    min_points : int
        最少的有效点数, 低于此值的块会被重新采样
    virtual_len : int
        一个 epoch 的虚拟样本数
    """

    def __init__(self, parsed_npz, pier_class=8, crop_radius=15.0,
                 min_points=500, virtual_len=200):
        data = np.load(parsed_npz)
        self.all_points = data["points"].astype(np.float32)
        all_labels_full = data["labels"].ravel()

        # 二值化标签: pier_class → 1, 其他 → 0
        self.all_labels = np.where(all_labels_full == pier_class, 1, 0).astype(np.int64)

        self.crop_radius = crop_radius
        self.min_points = min_points
        self.virtual_len = virtual_len

        # 预计算桥墩点的空间索引 (用于以桥墩为中心采样)
        self.pier_mask = self.all_labels == 1
        self.pier_pts = self.all_points[self.pier_mask]

        print(f"RealBridgeDataset: {len(self.all_points):,} 点 total, "
              f"{self.pier_mask.sum():,} pier (class {pier_class}), "
              f"crop={crop_radius}m")

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        # 以随机桥墩点为中心采样 (确保块内包含桥墩)
        # 如果桥墩点太少, 回退到全局随机采样
        while True:
            if len(self.pier_pts) > 0 and np.random.random() < 0.7:
                center = self.pier_pts[np.random.randint(len(self.pier_pts))]
            else:
                center = self.all_points[np.random.randint(len(self.all_points))]

            # 添加随机偏移, 避免每次都从同一点采样
            center = center + np.random.uniform(-5, 5, 3)

            # 半径筛选
            dists = np.linalg.norm(self.all_points - center[None, :], axis=1)
            crop_mask = dists < self.crop_radius

            if crop_mask.sum() < self.min_points:
                continue

            pts = self.all_points[crop_mask].copy()
            lbl = self.all_labels[crop_mask].copy()

            # 去中心化 (跟训练数据保持一致)
            pts -= center[None, :]

            # 限制点数 (避免 OOM)
            max_pts = 8000
            if len(pts) > max_pts:
                keep = np.random.choice(len(pts), max_pts, replace=False)
                pts = pts[keep]
                lbl = lbl[keep]

            return (torch.from_numpy(pts),
                    torch.from_numpy(lbl))
