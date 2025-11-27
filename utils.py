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
    vit_b_16, ViT_B_16_Weights,
    VisionTransformer
)


# ============================================================
# 1. Backbone 定义（统一接口）
# ============================================================

def build_backbone(backbone_type="resnet50", pretrained=False, image_size=None):
    """
    构建统一的 backbone，用于公平比较
    
    Args:
        backbone_type: "resnet50", "vit_b_16", "vit_s_14", "vit_b_14"
        pretrained: 是否使用预训练权重
        image_size: 图像尺寸（仅对 ViT 有效，用于支持非 224 的输入）
    
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
    
    elif backbone_type == "vit_s_16":
        # ViT-S/16: Small model with patch_size=16
        # 参数量: ~22M
        backbone = VisionTransformer(
            image_size=image_size or 224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=1000,  # 临时，后面会移除
            representation_size=None,
        )
        
        # 如果使用预训练权重，尝试从 DINOv2 加载（如果可用）
        if pretrained:
            try:
                # 尝试从 torch.hub 加载 DINOv2 ViT-S/16 预训练权重（如果有）
                # 注意：DINOv2 官方只有 S/14, B/14, L/14, g/14，没有 S/16
                # 这里尝试加载 S/14 并适配到 16
                dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                _copy_dinov2_weights(backbone, dinov2_model)
            except Exception as e:
                print(f"⚠️  无法加载 DINOv2 ViT-S/16 预训练权重: {e}")
                print("   将使用随机初始化的权重")
        
        feat_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()  # 移除分类头
        return backbone, feat_dim
    
    elif backbone_type == "vit_b_16":
        if pretrained:
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            backbone = vit_b_16(weights=None)
        
        patch_size = 16
        default_image_size = 224
        
        # 如果指定了 image_size 且不是 224，需要修改 ViT 的 image_size
        if image_size is not None and image_size != default_image_size:
            backbone = _adapt_vit_image_size(backbone, image_size, patch_size, pretrained)
        
        # ViT 的 CLS token 维度
        feat_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()  # 移除分类头
        return backbone, feat_dim
    
    elif backbone_type == "vit_s_14":
        # ViT-S/14: Small model with patch_size=14
        # 参数量: ~22M
        backbone = VisionTransformer(
            image_size=image_size or 224,
            patch_size=14,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=1000,  # 临时，后面会移除
            representation_size=None,
        )
        
        # 如果使用预训练权重，尝试从 DINOv2 加载（如果可用）
        if pretrained:
            try:
                # 尝试从 torch.hub 加载 DINOv2 ViT-S/14 预训练权重
                dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                # 复制权重（需要匹配的层）
                _copy_dinov2_weights(backbone, dinov2_model)
            except Exception as e:
                print(f"⚠️  无法加载 DINOv2 ViT-S/14 预训练权重: {e}")
                print("   将使用随机初始化的权重")
        
        feat_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()  # 移除分类头
        return backbone, feat_dim
    
    elif backbone_type == "vit_b_14":
        # ViT-B/14: Base model with patch_size=14
        # 参数量: ~86M
        backbone = VisionTransformer(
            image_size=image_size or 224,
            patch_size=14,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=1000,  # 临时，后面会移除
            representation_size=None,
        )
        
        # 如果使用预训练权重，尝试从 DINOv2 加载（如果可用）
        if pretrained:
            try:
                # 尝试从 torch.hub 加载 DINOv2 ViT-B/14 预训练权重
                dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                # 复制权重（需要匹配的层）
                _copy_dinov2_weights(backbone, dinov2_model)
            except Exception as e:
                print(f"⚠️  无法加载 DINOv2 ViT-B/14 预训练权重: {e}")
                print("   将使用随机初始化的权重")
        
        feat_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()  # 移除分类头
        return backbone, feat_dim
    
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}. Available: resnet50, vit_s_16, vit_b_16, vit_s_14, vit_b_14")


def _adapt_vit_image_size(backbone, image_size, patch_size, pretrained):
    """适配 ViT 到不同的图像尺寸"""
    backbone.image_size = image_size
    num_patches = (image_size // patch_size) ** 2
    
    # 更新位置编码的尺寸
    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'pos_embedding'):
        embed_dim = backbone.encoder.pos_embedding.shape[-1]
        new_pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)  # +1 for CLS token
        )
        # 如果使用预训练权重，尝试插值旧的位置编码
        if pretrained and backbone.encoder.pos_embedding.shape[1] > 1:
            old_num_patches = backbone.encoder.pos_embedding.shape[1] - 1
            old_pos = backbone.encoder.pos_embedding[:, 1:].reshape(
                1, int(old_num_patches ** 0.5), int(old_num_patches ** 0.5), embed_dim
            ).permute(0, 3, 1, 2)
            new_pos = nn.functional.interpolate(
                old_pos, 
                size=(int(num_patches ** 0.5), int(num_patches ** 0.5)), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1).reshape(1, num_patches, embed_dim)
            new_pos_embedding.data[:, 1:] = new_pos
            new_pos_embedding.data[:, 0] = backbone.encoder.pos_embedding.data[:, 0]  # CLS token
        backbone.encoder.pos_embedding = new_pos_embedding
    
    return backbone


def _copy_dinov2_weights(target_model, source_model):
    """从 DINOv2 模型复制权重到目标模型"""
    # DINOv2 模型的结构可能与标准 ViT 略有不同
    # 这里尝试匹配可用的层
    target_state = target_model.state_dict()
    source_state = source_model.state_dict()
    
    matched_keys = []
    for key in target_state.keys():
        if key in source_state:
            if target_state[key].shape == source_state[key].shape:
                target_state[key] = source_state[key]
                matched_keys.append(key)
    
    target_model.load_state_dict(target_state)
    print(f"✔ 成功加载 {len(matched_keys)}/{len(target_state)} 层权重")


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

