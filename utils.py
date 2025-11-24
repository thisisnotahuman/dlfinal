"""
通用工具模块 - 完全可以共用的部分
==================================================
包含：
1. Backbone 定义（统一接口，公平比较）
2. Optimizer & Learning Rate Scheduler
3. 其他通用工具函数
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights
)


# ============================================================
# 1. Backbone 定义（统一接口）
# ============================================================

def build_backbone(backbone_type="resnet50", pretrained=False):
    """
    构建统一的 backbone，用于公平比较
    
    Args:
        backbone_type: "resnet50" 或 "vit_b_16"
        pretrained: 是否使用预训练权重
    
    Returns:
        backbone: nn.Module，输出特征维度
        feat_dim: int，特征维度
    """
    if backbone_type == "resnet50":
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=None)
        
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # 移除分类头
        return backbone, feat_dim
    
    elif backbone_type == "vit_b_16":
        if pretrained:
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            backbone = vit_b_16(weights=None)
        
        # ViT 的 CLS token 维度
        feat_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()  # 移除分类头
        return backbone, feat_dim
    
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")


# ============================================================
# 2. Optimizer & Learning Rate Scheduler
# ============================================================

def build_optimizer(model, optimizer_type="adamw", lr=1e-3, weight_decay=1e-4, **kwargs):
    """
    构建优化器
    
    Args:
        model: 模型
        optimizer_type: "adamw" 或 "sgd"
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他优化器参数
    
    Returns:
        optimizer: torch.optim.Optimizer
    """
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def build_scheduler(optimizer, scheduler_type="cosine", T_max=100, warmup_epochs=0, **kwargs):
    """
    构建学习率调度器（支持 warmup）
    
    Args:
        optimizer: 优化器
        scheduler_type: "cosine" 或 "step"
        T_max: 总训练轮数（cosine）或 step_size（step）
        warmup_epochs: warmup 轮数
        **kwargs: 其他调度器参数
    
    Returns:
        scheduler: 学习率调度器
    """
    if scheduler_type == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, **kwargs
        )
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    
    # 如果有 warmup，包装成 WarmupScheduler
    if warmup_epochs > 0:
        return WarmupScheduler(base_scheduler, warmup_epochs=warmup_epochs)
    
    return base_scheduler


class WarmupScheduler:
    """Warmup 学习率调度器包装器"""
    
    def __init__(self, scheduler, warmup_epochs=10):
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in scheduler.optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup: 线性增长
            for param_group, base_lr in zip(self.scheduler.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            self.scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [group['lr'] for group in self.scheduler.optimizer.param_groups]


# ============================================================
# 3. 其他通用工具
# ============================================================

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(save_path, epoch, model, optimizer, scheduler=None, **kwargs):
    """保存 checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.scheduler.state_dict() if hasattr(scheduler, 'scheduler') else scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(load_path, model, optimizer=None, scheduler=None, device=None):
    """加载 checkpoint"""
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if hasattr(scheduler, 'scheduler'):
            scheduler.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint.get("epoch", 0), checkpoint
