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
import torch.nn.functional as F
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
    method="simclr",  # 新增：方法类型
    num_local_crops=0,  # 新增：local crops 数量（DINOv2 使用）
):
    """
    构建双视图或多视图增强
    
    输入： x = [B, 3, H, W]，float32 GPU，已经是 [0, 1] 范围（经过 ToTensor）
    输出： 
        - SimCLR: [B, 2, 3, H, W]
        - DINOv2: [B, 2+num_local_crops, 3, H, W]（前2个是global，后面是local）
    
    流程：
    1. 输入已经是 [0, 1] 范围（从 ms_transform 的 ToTensor 得到）
    2. 在 [0, 1] 范围上进行增强操作
    3. 最后 normalize 到目标分布
    
    DINOv2 官方 multi-crop 规范：
    - Global crops (2个): 尺寸 224, crop scale (0.4, 1.0), 用于 teacher + student
    - Local crops (6-10个): 尺寸 96, crop scale (0.05, 0.4), 仅用于 student
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
    # DINOv2 官方要求：
    # - Global view 1: blur p=1.0, solarize p=0
    # - Global view 2: blur p=0.0, solarize p=0.2
    # - Local crops: blur p=0.5, solarize p=0
    is_dinov2 = method.lower() in ["dino", "dinov2", "ibot"]
    
    if strength == "strong":
        if is_dinov2:
            # DINOv2 官方配置
            aug_cfg = dict(
                use_rrc=True,
                cj=(0.8, 0.8, 0.8, 0.2),
                p_cj=0.8,
                p_gray=0.2,
                blur_p1=1.0,  # Global view 1: blur p=1.0
                blur_p2=0.0,  # ✅ 修复：Global view 2: blur p=0.0（不是 0.1）
                solarize_p2=0.2,  # Global view 2: solarize p=0.2
                blur_p_local=0.5,  # ✅ 新增：Local crops: blur p=0.5
                solarize_p_local=0.0,  # ✅ 新增：Local crops: solarize p=0
            )
        else:
            # SimCLR 或其他方法的配置
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
        if is_dinov2:
            # DINOv2 官方配置（medium 强度时也遵循相同模式）
            aug_cfg = dict(
                use_rrc=True,
                cj=(0.4, 0.4, 0.3, 0.1),
                p_cj=0.7,
                p_gray=0.15,
                blur_p1=1.0,  # Global view 1: blur p=1.0
                blur_p2=0.0,  # ✅ 修复：Global view 2: blur p=0.0
                solarize_p2=0.2,  # Global view 2: solarize p=0.2
                blur_p_local=0.5,  # Local crops: blur p=0.5
                solarize_p_local=0.0,  # Local crops: solarize p=0
            )
        else:
            aug_cfg = dict(
                use_rrc=True,
                cj=(0.4, 0.4, 0.3, 0.1),
                p_cj=0.7,
                p_gray=0.15,
                blur_p1=0.6,
                blur_p2=0.1,
                solarize_p2=0.0,
            )
    else:  # weak
        if is_dinov2:
            # DINOv2 官方配置（weak 强度时也遵循相同模式）
            aug_cfg = dict(
                use_rrc=True,
                cj=(0.3, 0.3, 0.2, 0.05),
                p_cj=0.5,
                p_gray=0.1,
                blur_p1=1.0,  # Global view 1: blur p=1.0
                blur_p2=0.0,  # ✅ 修复：Global view 2: blur p=0.0
                solarize_p2=0.2,  # Global view 2: solarize p=0.2
                blur_p_local=0.5,  # Local crops: blur p=0.5
                solarize_p_local=0.0,  # Local crops: solarize p=0
            )
        else:
            aug_cfg = dict(
                use_rrc=True,
                cj=(0.3, 0.3, 0.2, 0.05),
                p_cj=0.5,
                p_gray=0.1,
                blur_p1=0.3,
                blur_p2=0.0,
                solarize_p2=0.0,
            )
    if is_dinov2:
        # DINOv2 官方规范（ImageNet 224x224）：
        # - Global crops (2个): 尺寸 224, crop scale (0.4, 1.0), 用于 teacher + student
        # - Local crops (6-10个): 尺寸 96, crop scale (0.05, 0.4), 仅用于 student
        # - 比例关系：Global 224 / Local 96 ≈ 2.33
        
        # ✅ 修复：根据原始图像尺寸自适应调整
        # 如果原始数据是 96x96，按比例：
        # - Global crops: 96x96（从 96x96 中 crop，scale (0.4, 1.0)）
        # - Local crops: 应该更小，比如 32x32 或 48x48（保持比例关系）
        global_crop_scale = (0.4, 1.0)  # ✅ DINOv2 官方 global crop scale
        
        # ✅ 修复：按实现说明，所有视图（包括local）应该统一尺寸为 img_size
        # 说明要求：所有视图都是 112×112（适配后），包括 local
        # 但 local 的 crop scale 更小 (0.1~0.5)，然后 resize 到统一尺寸
        # 这样既保持了多尺度信息，又统一了输入尺寸
        if img_size >= 224:
            # 标准 DINOv2 配置：local crop 96，但 resize 到 224
            local_crop_size = 96
        else:
            # 对于较小的 img_size（如 96），local crop 也 resize 到 img_size
            # 使用更小的 crop size 来 crop，然后 resize 到 img_size
            # 这样可以保持多尺度信息，同时统一输入尺寸
            local_crop_size = max(32, img_size // 3)  # 至少 32×32，然后 resize 到 img_size
        
        local_crop_scale = (0.05, 0.4)  # ✅ DINOv2 官方 local crop scale
        if num_local_crops == 0:
            num_local_crops = 6  # ✅ 修复：按实现说明，使用 6 个 local crops（不是 8 个）
        
        print(f"📐 DINOv2 crop 配置: Global={img_size}×{img_size}, Local crop={local_crop_size}×{local_crop_size}→resize到{img_size}×{img_size}（统一尺寸）")
    else:
        # SimCLR 或其他：使用原来的 scale
        global_crop_scale = (0.2, 1.0)
        local_crop_size = img_size // 3  # 不使用，但定义以避免错误
        local_crop_scale = (0.05, 0.2)
        num_local_crops = 0  # SimCLR 不使用 local crops
    
    # 构建增强 pipeline（在 [0, 1] 范围上操作）
    def build_global_pipeline(view_idx: int):
        """构建 global crop 的增强 pipeline"""
        ops = []
        
        if aug_cfg["use_rrc"]:
            # 使用正确的 crop scale
            ops.append(v2.RandomResizedCrop(img_size, scale=global_crop_scale))
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
    
    def build_local_pipeline():
        """构建 local crop 的增强 pipeline（更弱的增强）"""
        ops = []
        
        # Local crop: 使用更小的 scale
        ops.append(v2.RandomResizedCrop(local_crop_size, scale=local_crop_scale))
        ops.append(v2.RandomHorizontalFlip(0.5))
        
        # Local crops 使用更弱的颜色抖动
        if aug_cfg["p_cj"] > 0:
            cj_weak = tuple(x * 0.5 for x in aug_cfg["cj"])
            ops.append(v2.RandomApply(
                [v2.ColorJitter(*cj_weak)],
                p=aug_cfg["p_cj"] * 0.5
            ))
        
        # ✅ DINOv2 官方要求：Local crops 使用 blur p=0.5
        if is_dinov2 and "blur_p_local" in aug_cfg:
            blur_p_local = aug_cfg.get("blur_p_local", 0.5)
            if blur_p_local > 0:
                k = max(3, int(0.1 * local_crop_size) // 2 * 2 + 1)
                ops.append(v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))],
                    p=blur_p_local
                ))
        
        # ✅ DINOv2 官方要求：Local crops 不使用 solarize (p=0)
        # 已经在 aug_cfg 中设置为 0，不需要添加
        
        return v2.Compose(ops)
    
    pip_v1 = build_global_pipeline(0)
    pip_v2 = build_global_pipeline(1)
    
    if is_dinov2 and num_local_crops > 0:
        local_pip = build_local_pipeline()
    
    def apply(x):
        """
        Args:
            x: [B, 3, H, W]，[0, 1] 范围的 tensor（GPU）
        
        Returns:
            SimCLR: [B, 2, 3, H, W]（两个都是 img_size×img_size）
            DINOv2: [B, 2+num_local_crops, 3, H, W]（所有视图都是 img_size×img_size，统一尺寸）
        """
        device = x.device
        mean_base = base_mean.to(device)
        std_base = base_std.to(device)
        mean_custom = mean_t.to(device)
        std_custom = std_t.to(device)
        
        # 输入 x 已经是 [0, 1] 范围（从 ToTensor 得到）
        x = torch.clamp(x, 0.0, 1.0)
        
        views = []
        
        # 生成 2 个 global crops
        v1 = pip_v1(x)  # [B, 3, img_size, img_size]
        v2 = pip_v2(x)  # [B, 3, img_size, img_size]
        v1 = torch.clamp(v1, 0.0, 1.0)
        v2 = torch.clamp(v2, 0.0, 1.0)
        v1 = (v1 - mean_custom) / std_custom
        v2 = (v2 - mean_custom) / std_custom
        views.append(v1)
        views.append(v2)
        
        # DINOv2: 生成 local crops
        if is_dinov2 and num_local_crops > 0:
            for _ in range(num_local_crops):
                v_local = local_pip(x)  # [B, 3, local_crop_size, local_crop_size]
                v_local = torch.clamp(v_local, 0.0, 1.0)
                
                # ✅ 修复：按实现说明，所有视图（包括local）应该统一尺寸为 img_size
                # Local crop 从更小的区域 crop（scale 0.1~0.5），然后 resize 到 img_size
                # 这样既保持了多尺度信息，又统一了输入尺寸，便于 stack
                try:
                    v_local = F.interpolate(
                        v_local,  # [B, 3, local_crop_size, local_crop_size]
                        size=(img_size, img_size),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    )  # [B, 3, img_size, img_size]
                except TypeError:
                    # 如果 antialias 不支持，使用不带 antialias 的版本
                    v_local = F.interpolate(
                        v_local,
                        size=(img_size, img_size),
                        mode='bilinear',
                        align_corners=False
                    )
                v_local = (v_local - mean_custom) / std_custom
                views.append(v_local)  # [B, 3, img_size, img_size]（统一尺寸）
        
        # 堆叠所有 views（现在所有 views 都是统一尺寸，可以 stack）
        return torch.stack(views, dim=1)  # [B, 2+num_local_crops, 3, img_size, img_size]
    
    return apply


# ============================================================
# 2. 基础 resize + to_tensor（不 normalize，normalize 在增强后做）
# ============================================================

def ms_transform(img_size=96):
    """
    基础变换：resize + to_tensor，不 normalize
    
    ✅ 说明：如果原始数据是 96×96，而 img_size=112，会自动上采样到 112×112
    - v2.Resize(img_size) 会将图像 resize 到指定尺寸（上采样或下采样）
    - v2.CenterCrop(img_size) 确保输出尺寸为 img_size×img_size
    - 这样 112 可以被 patch_size=14 整除，patch grid 是 8×8，没有像素丢失
    """
    return v2.Compose([
        v2.Resize(img_size),  # 如果原始是 96×96，img_size=112，会上采样到 112×112
        v2.CenterCrop(img_size),  # 确保输出是 img_size×img_size
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
# 本地文件 Dataset（支持递归搜索子目录）
# =============================
class LocalImageDataset(Dataset):
    def __init__(self, root_dir, base_transform, recursive=True):
        """
        Args:
            root_dir: 图片文件夹路径
            base_transform: 基础变换
            recursive: 是否递归搜索子目录（默认 True）
        """
        self.paths = []
        self.base_transform = base_transform
        
        # 支持的图片格式
        extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
        
        if recursive:
            # 递归搜索所有子目录
            for ext in extensions:
                pattern = os.path.join(root_dir, "**", f"*.{ext}")
                self.paths.extend(glob(pattern, recursive=True))
        else:
            # 只搜索根目录
            for ext in extensions:
                pattern = os.path.join(root_dir, f"*.{ext}")
                self.paths.extend(glob(pattern, recursive=False))
        
        self.paths.sort()
        
        if len(self.paths) == 0:
            raise ValueError(
                f"未找到图片文件！\n"
                f"  搜索路径: {root_dir}\n"
                f"  递归搜索: {recursive}\n"
                f"  支持的格式: {extensions}\n"
                f"  请检查路径是否正确，或图片是否在子目录中（使用 recursive=True）"
            )

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
    dataset_type="local",
    dataset_name=None,
    dataset_root=None,
    img_size=96,
    batch_size=64,
    num_workers=8,
    train_sample=None,
    strength="strong",
    method="simclr",  # 新增：方法类型，用于选择正确的增强策略
    num_local_crops=0,  # 新增：local crops 数量（DINOv2 使用）
):
    """
    加载数据集（支持 HuggingFace 和本地文件）
    
    Args:
        dataset_type: "huggingface" 或 "local"
        dataset_name: HuggingFace 数据集名称（dataset_type="huggingface" 时使用）
        dataset_root: 本地图片文件夹路径（dataset_type="local" 时使用）
        img_size: 图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载线程数
        train_sample: 训练集子集大小
        strength: 增强强度
    
    Returns:
        train_loader, eval_loader, base_transform, two_view_aug
    """
    print("=" * 60)
    print("⚡ Loading dataset with GPU-optimized pipeline")
    print("=" * 60)

    rng = random.Random(42)

    # --------------------------------------------------------
    # 读取数据集
    # --------------------------------------------------------
    if dataset_type == "huggingface":
        if dataset_name is None:
            dataset_name = "tsbpp/fall2025_deeplearning"  # 默认值
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset(dataset_name)
        base_train = dataset["train"]
        print(f"✔ HF dataset loaded: {len(base_train)} images")
        
        # HuggingFace datasets 默认返回 PIL Image，无需额外设置格式
        
        # 使用原来的 DINODatasetGPU
        base_transform = ms_transform(img_size)
        train_dataset = DINODatasetGPU(base_train, base_transform)
        
    elif dataset_type == "local":
        if dataset_root is None:
            raise ValueError("dataset_root must be provided when dataset_type='local'")
        
        # 检查路径是否存在
        if not os.path.exists(dataset_root):
            raise ValueError(f"数据集路径不存在: {dataset_root}")
        if not os.path.isdir(dataset_root):
            raise ValueError(f"路径不是目录: {dataset_root}")
        
        print(f"✔ Loading local dataset from: {dataset_root}")
        base_transform = ms_transform(img_size)
        
        # 尝试递归搜索（默认）
        try:
            full_dataset = LocalImageDataset(dataset_root, base_transform, recursive=True)
        except ValueError as e:
            # 如果递归搜索失败，尝试非递归
            print(f"⚠️  递归搜索失败，尝试非递归搜索...")
            full_dataset = LocalImageDataset(dataset_root, base_transform, recursive=False)
        
        train_dataset = full_dataset
        print(f"✔ Found {len(full_dataset)} local images")
        
    elif dataset_type == "cifar10":
        from torchvision.datasets import CIFAR10
        base_train = CIFAR10("./data", train=True, download=True)
        print(f"✔ CIFAR10 loaded: {len(base_train)} images")
        base_transform = ms_transform(img_size)
        train_dataset = DINODatasetGPU(base_train, base_transform)
        
    elif dataset_type == "cifar100":
        from torchvision.datasets import CIFAR100
        base_train = CIFAR100("./data", train=True, download=True)
        print(f"✔ CIFAR100 loaded: {len(base_train)} images")
        base_transform = ms_transform(img_size)
        train_dataset = DINODatasetGPU(base_train, base_transform)
        
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # --------------------------------------------------------
    # train_sample 逻辑
    # --------------------------------------------------------
    if len(train_dataset) == 0:
        raise ValueError(
            f"数据集为空！找不到任何图片文件。\n"
            f"  请检查:\n"
            f"  1. 路径是否正确: {dataset_root if dataset_type == 'local' else dataset_name}\n"
            f"  2. 图片格式是否支持: jpg, jpeg, png\n"
            f"  3. 如果图片在子目录中，确保使用递归搜索"
        )
    
    if train_sample is not None:
        train_sample = int(train_sample)
        total = len(train_dataset)
        use_n = min(train_sample, total)
        idxs = rng.sample(range(total), use_n)
        train_dataset = Subset(train_dataset, idxs)
        print(f"✔ Using train subset: {use_n}/{total} images")
    else:
        print(f"✔ Using full train set: {len(train_dataset)} images")

    # --------------------------------------------------------
    # 构建 transform 和 augment
    # --------------------------------------------------------
    two_view_aug = build_two_view_augment(
        img_size, 
        strength,
        method=method,
        num_local_crops=num_local_crops
    )

    # --------------------------------------------------------
    # 创建 DataLoader（GPU augment + 高速线程）
    # --------------------------------------------------------
    # 优化配置：平衡 prefetch_factor 和 persistent_workers
    if num_workers <= 8:
        prefetch_factor = 4
        use_persistent_workers = True
    elif num_workers <= 16:
        prefetch_factor = 3
        use_persistent_workers = True
    else: # num_workers > 16
        prefetch_factor = 2
        use_persistent_workers = False # 太多worker可能导致内存问题，禁用persistent_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=use_persistent_workers,
        drop_last=True,
        collate_fn=collate_fn_cpu,
    )
    
    print(f"✔ DataLoader 配置: num_workers={num_workers}, prefetch_factor={prefetch_factor}, persistent_workers={use_persistent_workers}")

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
