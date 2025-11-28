"""
方法基类 - 定义自监督学习方法的统一接口
==================================================
每个具体方法（SimCLR, MoCo, BYOL, DINO, iBOT, MAE等）需要继承此类
并实现自己的核心逻辑：
1. build_head() - 构建 projection/prediction head
2. compute_loss() - 计算损失函数
3. forward() - 前向传播
4. 其他方法特定的机制（teacher, queue, mask等）
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseSSLMethod(nn.Module, ABC):
    """
    自监督学习方法基类
    
    每个方法需要实现：
    - build_head(): 构建 projection/prediction head
    - compute_loss(): 计算损失
    - forward(): 前向传播（可选，如果方法需要特殊处理）
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        config: Dict[str, Any]
    ):
        """
        Args:
            backbone: 编码器 backbone（统一接口）
            feat_dim: backbone 输出特征维度
            config: 方法特定的配置字典
        """
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.config = config
        
        # 构建方法特定的 head（projector/predictor/decoder等）
        self.head = self.build_head()
    
    @abstractmethod
    def build_head(self) -> nn.Module:
        """
        构建方法特定的 head
        
        例如：
        - SimCLR: MLP projector
        - BYOL: MLP projector + predictor
        - MAE: decoder
        
        Returns:
            head: nn.Module
        """
        raise NotImplementedError
    
    @abstractmethod
    def compute_loss(
        self,
        views: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失函数（方法的核心逻辑）
        
        Args:
            views: [B, num_views, 3, H, W] 或方法特定的输入格式
            **kwargs: 其他方法特定的参数
        
        Returns:
            loss: 标量损失值
            loss_dict: 损失分解字典（用于日志记录）
        """
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        默认前向传播：backbone + head
        
        如果方法需要特殊处理（如 MAE 的 mask），可以重写此方法
        
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            features: 方法特定的输出
        """
        h = self.backbone(x)
        
        # 处理 ViT 的 3D 输出：如果是 [B, num_patches+1, feat_dim]，取 CLS token
        # 这是为了兼容性，确保所有方法都能正确处理 ViT backbone
        if len(h.shape) == 3:
            # ViT 输出: [B, num_patches+1, feat_dim]，第 0 个是 CLS token
            h = h[:, 0]  # [B, feat_dim]
        
        if self.head is not None:
            return self.head(h)
        return h
    
    def get_views(self, images: torch.Tensor) -> torch.Tensor:
        """
        从输入图像生成 views（用于训练循环）
        
        默认实现：直接返回 images（假设 views 已经在 dataloader 中生成）
        如果方法需要特殊处理（如 MAE 的 mask），可以重写此方法
        
        Args:
            images: [B, num_views, 3, H, W] 或 [B, 3, H, W]
        
        Returns:
            views: 处理后的 views
        """
        return images
    
    def update_ema(self, *args, **kwargs):
        """
        更新 EMA（用于有 teacher 网络的方法，如 BYOL, DINO, iBOT）
        
        默认实现：空操作
        """
        pass
    
    def get_encoder(self) -> nn.Module:
        """
        获取编码器（用于评估时提取特征）
        
        Returns:
            encoder: backbone 或 backbone + head 的一部分
        """
        return self.backbone

