# 快速开始指南

## 📋 架构总结

### ✅ 已实现的方法（4个）

| 方法 | Backbone 支持 | 范式 | 状态 |
|------|-------------|------|------|
| **SimCLR** | ResNet-50, ViT-B/16 | 对比学习 (NT-Xent) | ✅ 完全实现 |
| **MoCo v2** | ResNet-50, ViT-B/16 | 对比学习 (Memory Queue) | ✅ 完全实现 |
| **DINO** | ResNet-50, ViT-B/16 | 自蒸馏 (Teacher + Centering) | ✅ 完全实现 |
| **iBOT** | ResNet-50, ViT-B/16 | Patch一致性 (Token-level对齐) | ✅ 完全实现 |
| **MAE** | ViT-B/16 仅 | 重建式 (Masked Autoencoder) | ✅ 完全实现 |

### ⚠️ 占位符方法（3个）

| 方法 | 状态 |
|------|------|
| **BYOL** | 框架已搭建，待实现具体逻辑 |
| **VICReg** | 框架已搭建，待实现具体逻辑 |

---

## 🚀 运行指令

### 基础训练命令

```bash
python train.py \
    --method <方法名> \
    --backbone_type <backbone类型> \
    --batch_size <批次大小> \
    --epochs <训练轮数> \
    --lr <学习率>
```

### 1. SimCLR

```bash
# ResNet-50
python train.py \
    --method simclr \
    --backbone_type resnet50 \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-3 \
    --temperature 0.5 \
    --save_dir ./checkpoints_simclr_resnet

# ViT-B/16
python train.py \
    --method simclr \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --save_dir ./checkpoints_simclr_vit
```

### 2. MoCo v2

```bash
# ResNet-50
python train.py \
    --method moco \
    --backbone_type resnet50 \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-3 \
    --temperature 0.2 \
    --save_dir ./checkpoints_moco_resnet

# ViT-B/16
python train.py \
    --method moco \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --save_dir ./checkpoints_moco_vit
```

### 3. DINO

```bash
# ResNet-50（只有 CLS 对齐）
python train.py \
    --method dino \
    --backbone_type resnet50 \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-3 \
    --warmup_epochs 10 \
    --save_dir ./checkpoints_dino_resnet

# ViT-B/16（完整 DINO）
python train.py \
    --method dino \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --warmup_epochs 10 \
    --save_dir ./checkpoints_dino_vit
```

### 4. iBOT (满血版本)

```bash
# ResNet-50（只有 CLS 对齐，无 patch token 对齐）
python train.py \
    --method ibot \
    --backbone_type resnet50 \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-3 \
    --save_dir ./checkpoints_ibot_resnet

# ViT-B/16（完整 iBOT：CLS + patch token 对齐）
python train.py \
    --method ibot \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --save_dir ./checkpoints_ibot_vit
```

### 5. MAE

```bash
# 仅支持 ViT-B/16
python train.py \
    --method mae \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --img_size 224 \
    --save_dir ./checkpoints_mae
```

---

## 📊 完整参数说明

### 必需参数

- `--method`: 方法名称 (`simclr`, `moco`, `dino`, `ibot`, `mae`)
- `--backbone_type`: Backbone 类型 (`resnet50`, `vit_b_16`)

### 数据参数

```bash
--dataset_type huggingface          # 数据集类型 (huggingface/cifar10/cifar100)
--dataset_name tsbpp/fall2025_deeplearning  # HuggingFace 数据集名称
--img_size 96                       # 图像尺寸
--num_workers 8                     # 数据加载线程数
--train_sample 50000                # 训练集子集大小（可选）
--aug_strength strong               # 增强强度 (strong/weak)
```

### 模型参数

```bash
--backbone_type resnet50            # Backbone 类型
--pretrained_backbone               # 使用预训练 backbone（可选）
--proj_hidden_dim 2048              # Projection head 隐藏层维度
--proj_output_dim 128               # Projection head 输出维度
--temperature 0.5                   # 温度参数（SimCLR/MoCo）
```

### 训练参数

```bash
--batch_size 128                    # 批次大小
--epochs 100                        # 训练轮数
--lr 1e-3                          # 学习率
--weight_decay 1e-4                 # 权重衰减
--optimizer_type adamw              # 优化器 (adamw/sgd)
--scheduler_type cosine             # 学习率调度器 (cosine/step)
--warmup_epochs 10                  # Warmup 轮数
--use_amp                           # 使用自动混合精度（默认开启）
```

### 保存和日志

```bash
--save_dir ./checkpoints            # 保存目录
--save_freq 1                       # 每 N 个 epoch 保存一次
--log_freq 100                      # 每 N 个 step 记录一次日志
```

---

## 💡 使用示例

### 快速测试（小数据集）

```bash
python train.py \
    --method simclr \
    --backbone_type resnet50 \
    --batch_size 32 \
    --epochs 10 \
    --train_sample 1000 \
    --save_dir ./test_checkpoints
```

### 完整训练（推荐配置）

```bash
# DINO + ViT-B/16（最佳性能）
python train.py \
    --method dino \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --warmup_epochs 10 \
    --img_size 224 \
    --save_dir ./checkpoints_dino_vit

# iBOT + ViT-B/16（满血版本）
python train.py \
    --method ibot \
    --backbone_type vit_b_16 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --img_size 224 \
    --save_dir ./checkpoints_ibot_vit
```

### 使用预训练 Backbone

```bash
python train.py \
    --method simclr \
    --backbone_type resnet50 \
    --pretrained_backbone \
    --lr 1e-4 \
    --epochs 50
```

---

## 📁 数据加载

- **HuggingFace 数据集**: 自动下载并缓存到 `~/.cache/huggingface/datasets/`
- **CIFAR 数据集**: 自动下载到 `./data/`
- **无需手动准备数据**，首次运行会自动下载

---

## 🔍 评估

训练完成后，可以使用 `eval.py` 进行评估：

```python
from eval import evaluate_model
from model import build_method
from utils import load_checkpoint

# 加载模型
method = build_method("simclr", backbone_type="resnet50")
load_checkpoint("checkpoints_simclr/best.pth", method)

# 评估（需要提供 train_loader 和 val_loader）
results = evaluate_model(
    method.get_encoder(),
    train_loader,
    val_loader,
    device,
    eval_method="knn"  # 或 "linear_probe"
)
```

---

## 📝 注意事项

1. **MAE 仅支持 ViT**: MAE 需要 patch 结构，只能使用 `--backbone_type vit_b_16`
2. **iBOT 的 patch token 对齐**: 仅在 ViT 时启用，ResNet 只有 CLS 对齐
3. **批次大小**: ViT 通常需要较小的 batch size（64），ResNet 可以用更大的（128+）
4. **图像尺寸**: ViT 通常使用 224，ResNet 可以使用 96 或 224
5. **数据增强**: 当前使用 strong 增强，适合 SimCLR/DINO/iBOT；MAE 使用 weak 增强

