"""
高性能 SimCLR / DINO 数据加载 + GPU 强增强
==================================================
特点：
- 全部增强在 GPU 上批处理（torchvision v2）
- 无 PIL、无 for-loop
- 支持 --train_sample 用于快速子集训练
- DataLoader 多线程 + prefetch 优化
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.v2 as v2
from torchvision.models import ResNeXt50_32X4D_Weights
from datasets import load_dataset
import random


# ============================================================
# 1. GPU SimCLR 双视图增强
# ============================================================

def build_two_view_augment(
    img_size: int,
    strength="strong",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    输入： x = [B, 3, H, W]，float32 GPU，已经是 [0, 1] 范围（经过 ToTensor）
    输出： [B, 2, 3, H, W]，normalize 后的图像
    
    流程：
    1. 输入已经是 [0, 1] 范围（从 ms_transform 的 ToTensor 得到）
    2. 在 [0, 1] 范围上进行增强操作
    3. 最后 normalize 到目标分布
    """
    
    if strength.lower() not in {"strong", "medium", "weak"}:
        raise ValueError(f"strength must be strong/medium/weak, got {strength!r}")
    
    strength = strength.lower()
    
    # 获取 ImageNet 的 mean/std（用于参考，但最终使用传入的 mean/std）
    weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    base_mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
    base_std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
    
    # 目标 normalize 的 mean/std
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    
    # 增强配置
    if strength == "strong":
        aug_cfg = dict(
            use_rrc=True,
            cj=(0.8, 0.8, 0.8, 0.2),
            p_cj=0.8,
            p_gray=0.2,
            blur_p1=1.0,
            blur_p2=0.1,
            solarize_p2=0.2,
        )
    elif strength == "medium":
        aug_cfg = dict(
            use_rrc=False,
            cj=(0.4, 0.4, 0.3, 0.1),
            p_cj=0.7,
            p_gray=0.15,
            blur_p1=0.6,
            blur_p2=0.1,
            solarize_p2=0.0,
        )
    else:  # weak
        aug_cfg = dict(
            use_rrc=False,
            cj=(0.3, 0.3, 0.2, 0.05),
            p_cj=0.5,
            p_gray=0.1,
            blur_p1=0.3,
            blur_p2=0.0,
            solarize_p2=0.0,
        )
    
    # 构建增强 pipeline（在 [0, 1] 范围上操作）
    def build_pipeline(view_idx: int):
        ops = []
        
        if aug_cfg["use_rrc"]:
            ops.append(v2.RandomResizedCrop(img_size, scale=(0.2, 1.0)))
        else:
            ops.extend([
                v2.Resize(img_size),
                v2.CenterCrop(img_size),
            ])
        
        ops.append(v2.RandomHorizontalFlip(0.5))
        
        # ColorJitter（带概率）
        if aug_cfg["p_cj"] > 0:
            ops.append(v2.RandomApply(
                [v2.ColorJitter(*aug_cfg["cj"])],
                p=aug_cfg["p_cj"]
            ))
        
        # Grayscale
        if aug_cfg["p_gray"] > 0:
            ops.append(v2.RandomGrayscale(aug_cfg["p_gray"]))
        
        # GaussianBlur
        blur_p = aug_cfg["blur_p1"] if view_idx == 0 else aug_cfg["blur_p2"]
        if blur_p > 0:
            k = max(3, int(0.1 * img_size) // 2 * 2 + 1)
            ops.append(v2.RandomApply(
                [v2.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))],
                p=blur_p
            ))
        
        # Solarize（仅 view2）
        if view_idx == 1 and aug_cfg["solarize_p2"] > 0:
            ops.append(v2.RandomApply(
                [v2.RandomSolarize(0.2)],
                p=aug_cfg["solarize_p2"]
            ))
        
        return v2.Compose(ops)
    
    pip_v1 = build_pipeline(0)
    pip_v2 = build_pipeline(1)
    
    def apply(x):
        """
        Args:
            x: [B, 3, H, W]，[0, 1] 范围的 tensor（GPU）
        
        Returns:
            [B, 2, 3, H, W]，normalize 后的两个视图
        """
        device = x.device
        mean_base = base_mean.to(device)
        std_base = base_std.to(device)
        mean_custom = mean_t.to(device)
        std_custom = std_t.to(device)
        
        # 输入 x 已经是 [0, 1] 范围（从 ToTensor 得到）
        # 如果之前被 normalize 过，需要先反 normalize（但当前流程不会 normalize）
        # 这里为了兼容性，先确保在 [0, 1] 范围
        x = torch.clamp(x, 0.0, 1.0)
        
        # 在 [0, 1] 范围上进行增强
        v1 = pip_v1(x)  # [B, 3, H, W]
        v2 = pip_v2(x)  # [B, 3, H, W]
        
        # 确保在 [0, 1] 范围（增强操作可能略微超出）
        v1 = torch.clamp(v1, 0.0, 1.0)
        v2 = torch.clamp(v2, 0.0, 1.0)
        
        # Normalize 到目标分布
        v1 = (v1 - mean_custom) / std_custom
        v2 = (v2 - mean_custom) / std_custom
        
        return torch.stack([v1, v2], dim=1)  # [B, 2, 3, H, W]
    
    return apply


# ============================================================
# 2. 基础 resize + to_tensor（不 normalize，normalize 在增强后做）
# ============================================================

def ms_transform(img_size=96):
    """基础变换：只做 resize 和 to_tensor，不 normalize"""
    return v2.Compose([
        v2.Resize(img_size),
        v2.CenterCrop(img_size),
        v2.ToTensor(),  # 转换为 [0, 1] 范围的 tensor
        # 注意：不在这里 normalize，normalize 在增强后做
    ])


# ============================================================
# 3. Dataset（读取图片 + 基础变换）
# ============================================================

class DINODatasetGPU(Dataset):

    def __init__(self, base_ds, base_transform):
        self.ds = base_ds
        self.base_transform = base_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]

        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")

        x = self.base_transform(img)   # [3,H,W] CPU Tensor
        return x


# ============================================================
# 4. collate_fn（批量 CPU 堆叠，GPU 增强在主进程进行）
# ============================================================

def collate_fn_cpu(batch):
    """
    在 DataLoader worker 中只做 CPU 操作，避免 CUDA 多进程问题
    GPU 增强将在主进程中进行
    """
    # 在 CPU 上堆叠 batch
    x = torch.stack(batch, dim=0)  # [B, 3, H, W] CPU
    return x


# ============================================================
# 5. DataLoader 主函数
# ============================================================

import os
import time
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torch
import random

# =============================
# 本地文件 Dataset
# =============================
class LocalImageDataset(Dataset):
    def __init__(self, root_dir, base_transform):
        self.paths = []
        for ext in ["jpg", "jpeg", "png"]:
            self.paths.extend(glob(os.path.join(root_dir, f"*.{ext}")))
        self.paths.sort()
        self.base_transform = base_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.base_transform(img)   # [3,H,W]
        return x


# =============================
# 高性能本地 DataLoader
# =============================
def load_dino_data(
    dataset_root="/content/drive/MyDrive/fall2025_data/images",
    img_size=96,
    batch_size=64,
    num_workers=8,
    train_sample=None,
    strength="strong",
    dataset_type="local"
):
    """
    dataset_root: 本地图片文件夹路径
    """

    print("=" * 60)
    print("⚡ Loading LOCAL dataset with GPU-optimized pipeline")
    print("=" * 60)

    rng = random.Random(42)

    # --------------------------------------------------------
    # 读取本地所有图片
    # --------------------------------------------------------
    img_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        img_paths.extend(glob(os.path.join(dataset_root, f"*.{ext}")))
    img_paths.sort()

    total_imgs = len(img_paths)
    print(f"✔ Found {total_imgs} local images")

    # --------------------------------------------------------
    # train_sample
    # --------------------------------------------------------
    if train_sample is not None:
        train_sample = int(train_sample)
        use_n = min(train_sample, total_imgs)
        idxs = rng.sample(range(total_imgs), use_n)
        img_paths_subset = [img_paths[i] for i in idxs]
        print(f"✔ Using subset: {use_n}/{total_imgs} images")
    else:
        img_paths_subset = img_paths
        idxs = list(range(total_imgs))
        print(f"✔ Using full local dataset: {total_imgs} images")

    # --------------------------------------------------------
    # transform + augment
    # --------------------------------------------------------
    base_transform = ms_transform(img_size)
    two_view_aug = build_two_view_augment(img_size, strength)

    # Dataset
    full_dataset = LocalImageDataset(dataset_root, base_transform)

    # Subset if needed
    if train_sample is not None:
        train_dataset = Subset(full_dataset, idxs)
    else:
        train_dataset = full_dataset

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
        collate_fn=collate_fn_cpu,
    )

    print(f"✔ Train loader created: {len(train_loader)} batches")
    print("=" * 60)

    return train_loader, None, base_transform, two_view_aug




# ============================================================
# 6. 测试
# ============================================================

if __name__ == "__main__":
    loader, _, _, _ = load_dino_data(
        img_size=96,
        batch_size=128,
        num_workers=8,
        train_sample=50000,   # 测试 sample
        strength="strong"
    )

    for batch in loader:
        print("Batch:", batch.shape)  # [B,2,3,96,96]
        break
