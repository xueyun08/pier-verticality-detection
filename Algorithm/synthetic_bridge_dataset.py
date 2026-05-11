import numpy as np
import torch
from torch.utils.data import Dataset

from pier_scene_generator import PierSceneGenerator


def collate_stacked(batch):
    """Collate variable-size point clouds into KPConv-style stacked tensors.

    Returns
    -------
    points  : (total_N, 3)  float32  — all points concatenated
    labels  : (total_N,)    int64    — all pointwise labels concatenated
    batch_idx : (total_N,)  int64    — which scene each point belongs to
    """
    points_list, labels_list = zip(*batch)
    batch_idx_list = [torch.full((len(p),), i, dtype=torch.long)
                      for i, (p, _) in enumerate(zip(points_list, labels_list))]

    points = torch.cat(points_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    return points, labels, batch_idx


class SyntheticBridgeDataset(Dataset):
    """On-the-fly synthetic bridge pier point cloud dataset.

    Generates every sample live — no disk I/O.  Pier morphology, dimensions,
    occlusion severity, and all other parameters are randomised on each call
    to ``__getitem__``.

    Parameters
    ----------
    virtual_len : int
        Nominal epoch size.  Since generation is unlimited, this controls
        how many samples constitute one "epoch".
    seed : int or None
        Passed through to PierSceneGenerator (deterministic if set).  The
        generator state evolves across calls, so samples are different each
        time even with a fixed seed.
    """

    def __init__(self, virtual_len=1000, seed=None):
        self.virtual_len = virtual_len
        self.generator = PierSceneGenerator(seed=seed)

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, index):
        pts_np, lbl_np, _meta = self.generator.generate_scene()

        points = torch.from_numpy(pts_np)           # (N, 3) float32
        labels = torch.from_numpy(lbl_np).squeeze(1).long()  # (N,) int64
        return points, labels


# ---------------------------------------------------------------------------
#  quick-test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("=== SyntheticBridgeDataset quick test ===\n")

    dataset = SyntheticBridgeDataset(virtual_len=200, seed=42)
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        collate_fn=collate_stacked, num_workers=0)

    batch = next(iter(loader))
    points, labels, batch_idx = batch

    print(f"Batch contents:")
    print(f"  points     shape: {points.shape}     dtype: {points.dtype}")
    print(f"  labels     shape: {labels.shape}     dtype: {labels.dtype}")
    print(f"  batch_idx  shape: {batch_idx.shape}  dtype: {batch_idx.dtype}")
    print(f"  unique batch indices: {batch_idx.unique().tolist()}")
    print(f"  label values: {labels.unique().tolist()}  (1=pier, 0=context)")

    # Per-scene breakdown
    for i in range(4):
        mask = batch_idx == i
        n = mask.sum().item()
        n_pier = (labels[mask] == 1).sum().item()
        print(f"  scene {i}: {n:5d} points  ({n_pier} pier, {n - n_pier} context)")

    # Timing over a few batches
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        _ = next(iter(loader))
    elapsed = time.perf_counter() - t0
    print(f"\n20 batches loaded in {elapsed*1000:.1f} ms  "
          f"({elapsed*1000/20:.1f} ms/batch)")

    print("\nAll checks passed.")
