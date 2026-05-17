"""
Phase 2 真实点云数据集 — 多场景版。

支持同时从多个标注场景中采样空间块。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RealBridgeDataset(Dataset):
    """从多个真实大场景中采样空间块, 用于 Phase 2 微调。"""

    def __init__(self, npz_paths, pier_class=8, crop_radius=8.0,
                 min_points=300, virtual_len=600, target_density=6.0):
        self.crop_radius = crop_radius
        self.min_points = min_points
        self.virtual_len = virtual_len
        self.target_density = target_density
        self.pier_class = pier_class

        # 加载所有场景, 保持独立
        self.scenes = []
        total_pier = 0
        total_bg = 0
        total_pts = 0

        for path in npz_paths:
            data = np.load(path)
            pts = data["points"].astype(np.float32)
            lbl_full = data["labels"].ravel()
            lbl_bin = np.where(lbl_full == pier_class, 1, 0).astype(np.int64)

            pier_mask = lbl_bin == 1
            scene = {
                "points": pts,
                "labels": lbl_bin,
                "pier_pts": pts[pier_mask],
                "bg_pts": pts[~pier_mask],
            }
            self.scenes.append(scene)
            total_pts += len(pts)
            total_pier += pier_mask.sum()
            total_bg += (~pier_mask).sum()

        self.n_scenes = len(self.scenes)
        print(f"RealBridgeDataset: {self.n_scenes} scenes, {total_pts:,} pts total, "
              f"{total_pier:,} pier (class {pier_class}), "
              f"{total_bg:,} bg, crop={crop_radius}m")

    def __len__(self):
        return self.virtual_len

    def _subsample_to_density(self, pts, lbl):
        """体素降采样到目标密度。"""
        if len(pts) <= 500:
            return pts, lbl
        area = (np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])) + 1.0
        target_n = int(area * self.target_density)
        target_n = max(500, min(target_n, 5000))
        if len(pts) <= target_n:
            return pts, lbl
        voxel = np.sqrt(area / target_n)
        voxel_idx = np.floor(pts / max(voxel, 0.05)).astype(np.int64)
        _, inv_map, counts = np.unique(voxel_idx, axis=0,
                                        return_inverse=True, return_counts=True)
        n_voxels = inv_map.max() + 1
        sub_pts = np.zeros((n_voxels, 3), dtype=np.float32)
        np.add.at(sub_pts, inv_map, pts)
        sub_pts /= np.maximum(counts[:, None], 1)
        sub_lbl = np.zeros(n_voxels, dtype=np.int64)
        for c in [0, 1]:
            counts_c = np.bincount(inv_map[lbl == c], minlength=n_voxels)
            sub_lbl = np.where(counts_c > (counts / 2), c, sub_lbl)
        return sub_pts, sub_lbl

    def __getitem__(self, idx):
        while True:
            # 随机选一个场景
            sc = self.scenes[np.random.randint(self.n_scenes)]

            r = np.random.random()
            if r < 0.4 and len(sc["pier_pts"]) > 0:
                center = sc["pier_pts"][np.random.randint(len(sc["pier_pts"]))]
                center = center + np.random.uniform(-2, 2, 3)
            elif r < 0.55 and len(sc["bg_pts"]) > 0:
                ref = sc["pier_pts"][np.random.randint(len(sc["pier_pts"]))]
                center = ref + np.random.uniform(-8, 8, 3)
            else:
                center = sc["points"][np.random.randint(len(sc["points"]))]

            dists = np.linalg.norm(sc["points"] - center[None, :], axis=1)
            crop_mask = dists < self.crop_radius

            if crop_mask.sum() < self.min_points:
                continue

            pts = sc["points"][crop_mask].copy()
            lbl = sc["labels"][crop_mask].copy()
            pts, lbl = self._subsample_to_density(pts, lbl)
            pts -= center[None, :]

            if len(pts) < self.min_points:
                continue

            return (torch.from_numpy(pts), torch.from_numpy(lbl))
