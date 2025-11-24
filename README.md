# 通用自监督学习框架

这是一个模块化的自监督学习框架，支持多种方法（SimCLR, MoCo, BYOL, DINO, iBOT, VICReg, MAE等）的统一训练和评估。

## 框架结构

### ✅ 完全可以共用的部分（公共模块）

1. **`load_data.py`** - Dataset & Dataloader
   - 读图路径、加载成 PIL / Tensor、normalize 到 [0,1] 等
   - train/pretrain 都是不看 label 的 dataloader
   - 对所有方法通用

2. **`utils.py`** - Backbone 定义、Optimizer & Scheduler
   - `build_backbone()`: 统一构建 ResNet-50 或 ViT-B/16
   - `build_optimizer()`: 构建 AdamW / SGD
   - `build_scheduler()`: 构建 cosine LR、warmup 等
   - 为了公平比较，尽量保持 backbone 相同

3. **`eval.py`** - Eval Pipeline
   - 从 frozen encoder 抽 feature → 建 k-NN feature bank → 在 eval 上算 accuracy
   - 或 linear probe
   - 对所有方法共用，只换 checkpoint

### 🟡 半通用部分（统一框架，方法传入配置）

1. **`augmentation.py`** - Augmentation & ViewMaker
   - `build_augment(config)`: 根据配置构建增强（strong/weak）
   - `ViewMaker`: 统一的视图生成器，支持 k 个 global views 和 n 个 local views
   - 每个方法在 config 里指定数量和强度

2. **`base_method.py`** - 方法基类
   - 定义统一接口：`build_head()`, `compute_loss()`, `forward()`
   - 每个方法继承此类并实现自己的逻辑

### 🔴 不能共用部分（每个方法自己实现）

1. **`model.py`** - 具体方法实现
   - **SimCLR**: NT-Xent 对比损失（已实现）
   - **MoCo**: memory queue 里的对比损失（占位符）
   - **BYOL**: student vs teacher 的 MSE / cosine regression（占位符）
   - **DINO**: student softmax 对齐 teacher softmax（占位符）
   - **iBOT**: CLS 对齐 + patch token 对齐（占位符）
   - **VICReg**: variance / invariance / covariance 三项正则（占位符）
   - **MAE**: 重建像素/patch 的 L2/L1 损失（占位符）

2. **方法特定的机制**
   - Teacher 网络（BYOL, DINO, iBOT）和 EMA 更新
   - 负样本 queue（MoCo）
   - Mask 生成器（MAE/iBOT）
   - Token 级对齐 vs 只对齐 CLS

## 文件说明

```
.
├── load_data.py          # 数据加载（保持不变）
├── utils.py              # 通用工具（backbone, optimizer, scheduler）
├── augmentation.py       # 增强配置和 ViewMaker
├── base_method.py        # 方法基类（定义接口）
├── model.py              # 具体方法实现（SimCLR 等）
├── train.py              # 通用训练循环
├── eval.py               # 通用评估 pipeline
└── README.md             # 本文件
```

## 使用方法

### 训练

```bash
# 训练 SimCLR
python train.py \
    --method simclr \
    --backbone_type resnet50 \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-3 \
    --save_dir ./checkpoints_simclr

# 使用 ViT backbone
python train.py \
    --method simclr \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100

# 使用预训练 backbone
python train.py \
    --method simclr \
    --pretrained_backbone \
    --lr 1e-4
```

### 评估

```python
from eval import evaluate_model
from model import build_method
from utils import load_checkpoint

# 加载模型
method = build_method("simclr", backbone_type="resnet50")
load_checkpoint("checkpoints_simclr/best.pth", method)

# 评估
results = evaluate_model(
    method.get_encoder(),
    train_loader,
    val_loader,
    device,
    eval_method="knn"  # 或 "linear_probe"
)
```

## 添加新方法

要添加新的自监督学习方法，只需：

1. 在 `model.py` 中继承 `BaseSSLMethod`
2. 实现 `build_head()` 和 `compute_loss()`
3. 如果需要，实现 `update_ema()`, `get_views()` 等方法

示例：

```python
class MyMethod(BaseSSLMethod):
    def build_head(self):
        # 实现你的 head
        return nn.Sequential(...)
    
    def compute_loss(self, views, **kwargs):
        # 实现你的损失函数
        loss = ...
        return loss, {"loss": loss.item()}
```

然后在 `build_method()` 函数中注册你的方法即可。

## 设计原则

1. **完全通用**：dataloader / backbone / optimizer / eval 都做成完全通用，只换"方法模块"
2. **半通用**：augmentation / view maker / head 有统一框架，方法传入配置
3. **方法特定**：loss / teacher / queue / mask 等核心逻辑每个方法自己实现

这样的设计使得：
- 添加新方法只需实现核心逻辑
- 不同方法可以公平比较（相同 backbone、optimizer、eval）
- 代码复用性高，维护简单

## 下载test set
```python
python prepare_cub200_for_kaggle.py --download_dir ./raw_data --output_dir ./data

```
