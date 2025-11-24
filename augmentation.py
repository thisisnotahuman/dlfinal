"""
增强配置和 ViewMaker - 半通用部分
==================================================
统一的增强框架，每个方法通过配置传入不同参数
"""

import torch
import torchvision.transforms.v2 as v2
from typing import Dict, Any, Callable


# ============================================================
# 1. 增强配置构建器
# ============================================================

def build_augment(config: Dict[str, Any]) -> Callable:
    """
    根据配置构建增强函数
    
    Args:
        config: 增强配置字典，包含：
            - img_size: 图像尺寸
            - strength: "strong" (SimCLR/DINO/iBOT) 或 "weak" (MAE)
            - color_jitter: 颜色抖动强度
            - blur_prob: 模糊概率
            - solarize_prob: 曝光概率
    
    Returns:
        augment_fn: 增强函数，输入 [B, 3, H, W]，输出增强后的图像
    """
    img_size = config.get("img_size", 224)
    strength = config.get("strength", "strong")
    
    if strength == "strong":
        # SimCLR / DINO / iBOT: 强增强
        color_jitter = config.get("color_jitter", (0.8, 0.8, 0.8, 0.2))
        blur_prob = config.get("blur_prob", 1.0)
        solarize_prob = config.get("solarize_prob", 0.2)
        
        transforms = [
            v2.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            v2.RandomHorizontalFlip(0.5),
            v2.ColorJitter(*color_jitter),
            v2.RandomGrayscale(0.2),
        ]
        
        if blur_prob > 0:
            transforms.append(v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)))
        
        if solarize_prob > 0:
            transforms.append(v2.RandomSolarize(solarize_prob))
    
    elif strength == "weak":
        # MAE: 弱增强（重建任务）
        transforms = [
            v2.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(0.5),
        ]
    
    else:
        raise ValueError(f"Unknown strength: {strength}")
    
    augment_fn = v2.Compose(transforms)
    return augment_fn


# ============================================================
# 2. ViewMaker - 多视角生成器
# ============================================================

class ViewMaker:
    """
    统一的 ViewMaker：支持生成 k 个 global views 和 n 个 local views
    
    每个方法在 config 里指定数量：
    - SimCLR: 2 global views
    - DINO/iBOT/SwAV: 2 global + 多个 local views
    - MAE: 1 view + 内部 mask（不需要多份 crop）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，包含：
                - img_size: 图像尺寸
                - num_global_views: global views 数量（默认 2）
                - num_local_views: local views 数量（默认 0）
                - global_crop_scale: global crop 尺度范围（默认 (0.2, 1.0)）
                - local_crop_scale: local crop 尺度范围（默认 (0.05, 0.2)）
                - local_crop_size: local crop 尺寸（默认 img_size // 3）
                - augment_config: 增强配置（传给 build_augment）
        """
        self.img_size = config.get("img_size", 224)
        self.num_global_views = config.get("num_global_views", 2)
        self.num_local_views = config.get("num_local_views", 0)
        self.global_crop_scale = config.get("global_crop_scale", (0.2, 1.0))
        self.local_crop_scale = config.get("local_crop_scale", (0.05, 0.2))
        self.local_crop_size = config.get("local_crop_size", self.img_size // 3)
        
        # 构建增强函数
        augment_config = config.get("augment_config", {"img_size": self.img_size, "strength": "strong"})
        self.augment_fn = build_augment(augment_config)
        
        # 为 local views 构建单独的增强（通常更弱）
        local_augment_config = config.get("local_augment_config", augment_config.copy())
        local_augment_config["img_size"] = self.local_crop_size
        self.local_augment_fn = build_augment(local_augment_config)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成多个 views
        
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            views: [B, num_views, 3, H, W] 或 [B, num_views, 3, H_local, W_local]
                  如果只有 global views，返回 [B, num_global_views, 3, img_size, img_size]
                  如果有 local views，需要分别处理
        """
        batch_size = x.size(0)
        views = []
        
        # 生成 global views
        for _ in range(self.num_global_views):
            # 先 crop，再增强
            view = v2.RandomResizedCrop(
                self.img_size, 
                scale=self.global_crop_scale
            )(x)
            view = self.augment_fn(view)
            views.append(view)
        
        # 生成 local views
        for _ in range(self.num_local_views):
            view = v2.RandomResizedCrop(
                self.local_crop_size,
                scale=self.local_crop_scale
            )(x)
            view = self.local_augment_fn(view)
            views.append(view)
        
        # 堆叠: [B, num_views, 3, H, W]
        if len(views) > 0:
            return torch.stack(views, dim=1)
        else:
            return x.unsqueeze(1)  # [B, 1, 3, H, W]


# ============================================================
# 3. 便捷配置预设
# ============================================================

def get_simclr_augment_config(img_size=224):
    """SimCLR 增强配置：2 个 global views"""
    return {
        "img_size": img_size,
        "num_global_views": 2,
        "num_local_views": 0,
        "augment_config": {
            "img_size": img_size,
            "strength": "strong",
            "color_jitter": (0.8, 0.8, 0.8, 0.2),
            "blur_prob": 1.0,
            "solarize_prob": 0.2,
        }
    }


def get_dino_augment_config(img_size=224, num_local=6):
    """DINO/iBOT 增强配置：2 global + 多个 local views"""
    return {
        "img_size": img_size,
        "num_global_views": 2,
        "num_local_views": num_local,
        "augment_config": {
            "img_size": img_size,
            "strength": "strong",
            "color_jitter": (0.8, 0.8, 0.8, 0.2),
            "blur_prob": 1.0,
            "solarize_prob": 0.2,
        }
    }


def get_mae_augment_config(img_size=224):
    """MAE 增强配置：1 个 view（弱增强，mask 在模型内部处理）"""
    return {
        "img_size": img_size,
        "num_global_views": 1,
        "num_local_views": 0,
        "augment_config": {
            "img_size": img_size,
            "strength": "weak",
        }
    }

