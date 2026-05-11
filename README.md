# Pier Verticality Detection — Sim2Real 桥墩点云分割

基于 KPConv 的 Sim2Real 桥墩语义分割项目。通过程序化生成海量随机桥墩点云进行预训练，再迁移到真实点云微调，解决桥墩检测任务中真实标注数据稀缺的问题。

## 整体流程

```
┌──────────────────────────────────────────────────────────┐
│  Phase 1: 海量仿真预训练 (On-the-fly, 零磁盘IO)            │
│                                                          │
│  PierSceneGenerator  ──→  DegradationEngine              │
│  (多形态桥墩+上下文)       (噪声/遮挡/倾斜/密度衰减)         │
│         │                        │                       │
│         └────────┬───────────────┘                       │
│                  ▼                                       │
│     SyntheticBridgeDataset                               │
│     (torch.utils.data.Dataset)                           │
│                  │                                       │
│                  ▼                                       │
│           KPConvUNet  ←  MixedLoss (Focal + Dice)        │
│                  │                                       │
│                  ▼                                       │
│           Phase 1 预训练权重                               │
└──────────────────────────────────────────────────────────┘
                         │
                         │  LR × 0.1
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 2: 真实域微调                                      │
│                                                          │
│  手工标注 .pcd 真实点云  ──→  KPConvUNet (fine-tune)        │
└──────────────────────────────────────────────────────────┘
```

## 项目结构

```
Algorithm/
├── pier_scene_generator.py       # 桥墩场景程序化生成器
├── degradation_engine.py         # 物理退化引擎 (领域随机化)
├── synthetic_bridge_dataset.py   # PyTorch 即时数据流 Dataset
├── kpconv_model.py               # KPConv U-Net 分割网络 + 混合损失
├── train.py                      # 两阶段训练主循环
└── Prompt/                       # 设计文档
    ├── 1.多形态桥墩与上下文生成器.docx
    ├── 2.剧毒物理退化引擎.docx
    ├── 3.PyTorch 即时数据流.docx
    └── 4.KPConv 网络封装与 Sim2Real 训练主循环.docx
```

## 核心模块

### PierSceneGenerator — 桥墩场景生成器

程序化生成三种桥墩形态及其工程上下文：

| 桥墩类型 | 说明 | 几何模型 |
|---------|------|---------|
| 圆柱墩 | 标准圆柱形桥墩 | 圆柱体表面采样 |
| 重力式墩 | 矩形/梯形截面实体墩 | 截顶矩形金字塔表面采样 |
| Y 型墩 | 带分叉的 V 形桥墩 | 主柱 + 两支倾斜分支圆柱 |

每个场景自动附带：
- **桥面 (主梁)**：水平宽大方盒，架设在桥墩顶部
- **地面/水面**：水平平面，位于桥墩基底
- **硬负样本**：细长圆柱（路灯/脚手架），随机分布于桥墩周围

```python
from pier_scene_generator import PierSceneGenerator

gen = PierSceneGenerator(seed=42)
points, labels, meta = gen.generate_scene()
# points: (N, 3) float32  — 点云坐标
# labels: (N, 1) int8     — 1=桥墩, 0=背景
# meta:   dict            — 各组件参数
```

### DegradationEngine — 物理退化引擎

四类极端退化，实现 Sim2Real 领域随机化：

| 退化类型 | 效果 | 参数范围 |
|---------|------|---------|
| `apply_sensor_noise` | 高斯散斑噪声 (XYZ) | σ = 1–5 cm |
| `apply_z_density_decay` | 远距离点随机丢弃 (模拟 LiDAR 衰减) | 最高丢弃 15–55% |
| `apply_occlusion_holes` | 球形遮挡 (模拟植被) | 1–8 个球, 半径 0.2–2.0 m |
| `apply_random_tilt` | 微小随机旋转 (模拟施工偏差) | 0–3° |

### SyntheticBridgeDataset — 即时数据流

继承 `torch.utils.data.Dataset`，在 `__getitem__` 中实时生成场景，零磁盘 I/O。每次访问都是全新随机场景，天然抗过拟合。

```python
from synthetic_bridge_dataset import SyntheticBridgeDataset, collate_stacked
from torch.utils.data import DataLoader

dataset = SyntheticBridgeDataset(virtual_len=1000, seed=42)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_stacked)

points, labels, batch_idx = next(iter(loader))
# points:    (total_N, 3)  float32  拼接后的所有场景点云
# labels:    (total_N,)    int64    逐点标签
# batch_idx: (total_N,)    int64    每个点属于哪个场景 (0–3)
```

### KPConvUNet — 分割网络

三层 U-Net 架构的刚性 KPConv 网络：

- **编码器**：3 级体素降采样 (0.5 m → 1.0 m)，感受野逐步扩大
- **解码器**：2 级最近邻上采样 + 跳跃连接
- **核点**：Fibonacci 球面均匀分布 15 个核点
- **参数量**：~1.87 M

### MixedLoss — 混合损失

```
L = 0.5 × DiceLoss + 0.5 × FocalLoss(α=0.25, γ=2.0)
```

- **Focal Loss**：降低易分类样本权重，聚焦难样本
- **Dice Loss**：直接优化前景 IoU，缓解类别不平衡

## 环境依赖

```
Python >= 3.10
torch >= 2.0  (CUDA 推荐)
numpy
scipy
```

安装：

```bash
pip install torch numpy scipy
```

## 使用方法

### 快速测试

验证完整管线是否正常工作：

```bash
python Algorithm/train.py --quick_test
```

### Phase 1 仿真预训练

```bash
python Algorithm/train.py --phase1_epochs 100 --batch_size 2 --save_dir ./checkpoints
```

关键参数：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--phase1_epochs` | 100 | 预训练轮数 |
| `--phase1_lr` | 1e-2 | 初始学习率 (Cosine 退火) |
| `--batch_size` | 4 | 批大小 (GPU 显存不足时减小) |
| `--virtual_len` | 500 | 每 Epoch 虚拟样本数 |
| `--save_dir` | ./checkpoints | 模型保存目录 |

### Phase 2 真实域微调

```bash
python Algorithm/train.py \
  --phase1_epochs 100 \
  --phase2_epochs 20 \
  --phase2_lr 1e-3 \
  --real_data_dir ./real_pcd \
  --save_dir ./checkpoints
```

> Phase 2 需要 `.pcd` 格式的真实标注点云，接口已在 `build_real_loader()` 中预留 (TODO 标记)。

## 性能参考

| 指标 | 数值 |
|------|------|
| 场景生成 (含退化) | ~3 ms/scene |
| 单 Batch 加载 (bs=4) | ~11 ms |
| 模型参数量 | 1.87 M |
| GPU 显存占用 (bs=2) | ~3.5 GB |
| 5 Batch 训练后 IoU | > 0.90 (合成数据) |

> 测试环境：RTX 3050 Laptop GPU (4 GB VRAM), torch 2.6.0+cu124

## License

MIT
