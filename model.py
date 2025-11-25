model1.py
"""
具体方法实现 - 每个自监督学习方法的具体实现
==================================================
目前包含：
- SimCLR: NT-Xent 对比损失
- 其他方法（MoCo, BYOL, DINO, iBOT, MAE等）可以在这里添加
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from base_method import BaseSSLMethod
from utils import build_backbone


# ============================================================
# SimCLR 实现
# ============================================================

class SimCLR(BaseSSLMethod):
    """
    SimCLR: 简单的对比学习
    - Loss: NT-Xent 对比损失（正样本拉近，大量负样本推远）
    - Head: MLP projector
    - Views: 2 个 global views
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        config: Dict[str, Any]
    ):
        super().__init__(backbone, feat_dim, config)
        self.temperature = config.get("temperature", 0.5)
    
    def build_head(self) -> nn.Module:
        """构建 MLP projection head"""
        proj_hidden_dim = self.config.get("proj_hidden_dim", 2048)
        proj_output_dim = self.config.get("proj_output_dim", 128)
        
        projector = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim)
        )
        return projector
    
    def compute_loss(
        self,
        views: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 NT-Xent 对比损失
        
        Args:
            views: [B, 2, 3, H, W] 两个增强视图
        
        Returns:
            loss: 标量损失值
            loss_dict: 损失分解字典
        """
        # 分离两个视图
        x_i = views[:, 0]  # [B, 3, H, W]
        x_j = views[:, 1]  # [B, 3, H, W]
        
        # 前向传播
        z_i = self.forward(x_i)  # [B, proj_dim]
        z_j = self.forward(x_j)  # [B, proj_dim]
        
        # 归一化（如果 head 没有归一化）
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 计算 NT-Xent 损失
        loss = self._nt_xent(z_i, z_j, self.temperature)
        
        return loss, {"loss": loss.item()}
    
    def _nt_xent(self, z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失
        
        Args:
            z_i: [B, d] 第一个视图的特征
            z_j: [B, d] 第二个视图的特征
            temperature: 温度参数
        
        Returns:
            loss: 标量损失值
        """
        batch_size = z_i.size(0)
        
        # 拼接所有特征: [2B, d]
        z = torch.cat([z_i, z_j], dim=0)
        
        # 计算相似度矩阵: [2B, 2B]
        sim = torch.matmul(z, z.T) / temperature
        
        # 掩码对角线（自己与自己的相似度）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e4)  # ⬅️ 改成 -1e4（float16安全范围）
        
        # 正样本索引：第 i 个样本的正样本是第 i+B 个样本
        pos_idx = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (2 * batch_size)
        
        # 交叉熵损失
        loss = F.cross_entropy(sim, pos_idx)
        
        return loss


# ============================================================
# MoCo v2 实现
# ============================================================

class MoCo(BaseSSLMethod):
    """
    MoCo v2: Momentum Contrast
    - Loss: memory queue 里的对比损失
    - 机制: momentum encoder + memory queue
    - Backbone: ResNet-50 或 ViT-B/16
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        config: Dict[str, Any]
    ):
        super().__init__(backbone, feat_dim, config)
        
        # MoCo 参数
        self.momentum = config.get("momentum", 0.999)
        self.temperature = config.get("temperature", 0.2)
        self.queue_size = config.get("queue_size", 65536)
        
        # 创建 momentum encoder（不更新梯度）
        self.momentum_backbone = self._build_momentum_encoder()
        self.momentum_head = self._build_momentum_head()
        
        # 初始化 momentum encoder
        self._init_momentum_encoder()
        
        # Memory queue
        self.register_buffer("queue", torch.randn(self.queue_size, self.config.get("proj_output_dim", 128)))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _build_momentum_encoder(self) -> nn.Module:
        """构建 momentum encoder（student backbone 的副本）"""
        import copy
        momentum_backbone = copy.deepcopy(self.backbone)
        # 冻结参数（不参与梯度更新）
        for param in momentum_backbone.parameters():
            param.requires_grad = False
        return momentum_backbone
    
    def _build_momentum_head(self) -> nn.Module:
        """构建 momentum head（student head 的副本）"""
        import copy
        momentum_head = copy.deepcopy(self.head)
        # 冻结参数
        for param in momentum_head.parameters():
            param.requires_grad = False
        return momentum_head
    
    def _init_momentum_encoder(self):
        """初始化 momentum encoder 与 student 相同"""
        for param_q, param_k in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.head.parameters(), self.momentum_head.parameters()):
            param_k.data.copy_(param_q.data)
    
    def build_head(self) -> nn.Module:
        """构建 MLP projection head"""
        proj_hidden_dim = self.config.get("proj_hidden_dim", 2048)
        proj_output_dim = self.config.get("proj_output_dim", 128)
        
        projector = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim)
        )
        return projector
    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """EMA 更新 momentum encoder"""
        for param_q, param_k in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.head.parameters(), self.momentum_head.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新 memory queue"""
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        
        # 替换 queue 中的旧特征
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # 处理队列溢出
            self.queue[ptr:] = keys[:self.queue_size - ptr]
            self.queue[:batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:]
            ptr = batch_size - (self.queue_size - ptr)
        
        self.queue_ptr[0] = ptr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoCo 前向传播（支持 ViT）
        
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            features: [B, proj_dim] 投影后的特征
        """
        h = self.backbone(x)
        
        # 如果是 ViT，取 CLS token
        if len(h.shape) == 3:
            h = h[:, 0]  # [B, feat_dim] CLS token
        
        if self.head is not None:
            return self.head(h)
        return h
    
    def compute_loss(
        self,
        views: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 MoCo 对比损失
        
        Args:
            views: [B, 2, 3, H, W] 两个增强视图
        
        Returns:
            loss: 标量损失值
            loss_dict: 损失分解字典
        """
        # 分离两个视图
        x_q = views[:, 0]  # query: [B, 3, H, W]
        x_k = views[:, 1]  # key: [B, 3, H, W]
        
        # Student 前向传播（query）
        q = self.forward(x_q)  # [B, proj_dim]
        q = F.normalize(q, dim=1)
        
        # Momentum encoder 前向传播（key）
        with torch.no_grad():
            h_k = self.momentum_backbone(x_k)
            # 如果是 ViT，取 CLS token
            if len(h_k.shape) == 3:
                h_k = h_k[:, 0]  # [B, feat_dim] CLS token
            k = self.momentum_head(h_k)
            k = F.normalize(k, dim=1)
        
        # 计算正样本相似度: [B]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B, 1]
        
        # 从 queue 中获取负样本: [B, queue_size, proj_dim]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach().T])  # [B, queue_size]
        
        # 拼接正负样本: [B, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # 标签：正样本是第 0 个
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        # 更新 queue
        self._dequeue_and_enqueue(k)
        
        return loss, {"loss": loss.item()}
    
    def update_ema(self):
        """更新 momentum encoder"""
        self._update_momentum_encoder()


class BYOL(BaseSSLMethod):
    """
    BYOL: Bootstrap Your Own Latent
    - Loss: student vs teacher 的 MSE / cosine regression
    - 需要实现: teacher 网络和 EMA 更新
    """
    
    def build_head(self) -> nn.Module:
        # TODO: 实现 BYOL 的 projector + predictor
        raise NotImplementedError("BYOL 尚未实现")
    
    def compute_loss(self, views: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        # TODO: 实现 BYOL 的损失（需要 teacher 网络）
        raise NotImplementedError("BYOL 尚未实现")
    
    def update_ema(self, *args, **kwargs):
        # TODO: 实现 EMA 更新
        raise NotImplementedError("BYOL 尚未实现")


class DINO(BaseSSLMethod):
    """
    DINO: Knowledge Distillation with No Labels
    - Loss: student softmax 对齐 teacher softmax（有 temperature + centering）
    - 机制: teacher 网络、centering、EMA 更新、multi-crop
    - Backbone: ResNet-50 或 ViT-B/16
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        config: Dict[str, Any]
    ):
        super().__init__(backbone, feat_dim, config)
        
        # DINO 参数
        self.momentum = config.get("momentum", 0.996)
        self.temperature = config.get("temperature", 0.07)
        self.temperature_teacher = config.get("temperature_teacher", 0.04)
        self.center_momentum = config.get("center_momentum", 0.9)
        self.warmup_teacher_temp = config.get("warmup_teacher_temp", 0.04)
        self.warmup_teacher_temp_epochs = config.get("warmup_teacher_temp_epochs", 30)
        
        # 创建 teacher 网络
        self.teacher_backbone = self._build_teacher_encoder()
        self.teacher_head = self._build_teacher_head()
        self._init_teacher()
        
        # Centering
        self.register_buffer("center", torch.zeros(1, self.config.get("proj_output_dim", 65536)))
        self.current_epoch = 0
    
    def _build_teacher_encoder(self) -> nn.Module:
        """构建 teacher encoder"""
        import copy
        teacher = copy.deepcopy(self.backbone)
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher
    
    def _build_teacher_head(self) -> nn.Module:
        """构建 teacher head"""
        import copy
        teacher_head = copy.deepcopy(self.head)
        for param in teacher_head.parameters():
            param.requires_grad = False
        return teacher_head
    
    def _init_teacher(self):
        """初始化 teacher 与 student 相同"""
        for param_q, param_k in zip(self.backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.head.parameters(), self.teacher_head.parameters()):
            param_k.data.copy_(param_q.data)
    
    def build_head(self) -> nn.Module:
        """构建 DINO head（MLP projector）"""
        proj_hidden_dim = self.config.get("proj_hidden_dim", 2048)
        proj_output_dim = self.config.get("proj_output_dim", 65536)  # DINO 使用更大的输出维度
        
        projector = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.LayerNorm(proj_output_dim, eps=1e-6)
        )
        return projector
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DINO 前向传播：只对齐 CLS token"""
        # ViT 输出: [B, num_patches + 1, feat_dim] 或 [B, feat_dim]
        h = self.backbone(x)
        
        # 如果是 ViT，取 CLS token
        if len(h.shape) == 3:
            h = h[:, 0]  # [B, feat_dim] CLS token
        
        if self.head is not None:
            return self.head(h)
        return h
    
    @torch.no_grad()
    def _update_teacher(self):
        """EMA 更新 teacher"""
        for param_q, param_k in zip(self.backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.head.parameters(), self.teacher_head.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        """更新 centering"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def _get_teacher_temp(self):
        """获取 teacher temperature（带 warmup）"""
        if self.current_epoch < self.warmup_teacher_temp_epochs:
            return self.warmup_teacher_temp + (self.temperature_teacher - self.warmup_teacher_temp) * \
                   (self.current_epoch / self.warmup_teacher_temp_epochs)
        return self.temperature_teacher
    
    def compute_loss(
        self,
        views: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 DINO 损失（multi-crop 支持）
        
        Args:
            views: [B, num_views, 3, H, W] 多个视图（2 global + N local）
        
        Returns:
            loss: 标量损失值
            loss_dict: 损失分解字典
        """
        # 分离 global 和 local views
        # 假设前 2 个是 global views，后面是 local views
        num_global = 2
        global_views = views[:, :num_global]  # [B, 2, 3, H, W]
        local_views = views[:, num_global:] if views.size(1) > num_global else None  # [B, N_local, 3, H, W] 或 None
        
        # Student 处理所有 views
        student_outputs = []
        for i in range(views.size(1)):
            x = views[:, i]  # [B, 3, H, W]
            out = self.forward(x)
            student_outputs.append(out)
        
        # Teacher 只处理 global views
        teacher_outputs = []
        with torch.no_grad():
            for i in range(num_global):
                x = global_views[:, i]  # [B, 3, H, W]
                h = self.teacher_backbone(x)
                if len(h.shape) == 3:
                    h = h[:, 0]  # CLS token
                out = self.teacher_head(h)
                teacher_outputs.append(out)
        
        # 计算损失
        total_loss = 0
        num_losses = 0
        
        teacher_temp = self._get_teacher_temp()
        
        # Student 的每个输出与 teacher 的每个 global view 对齐
        for student_out in student_outputs:
            student_out = F.normalize(student_out, dim=1)
            
            for teacher_out in teacher_outputs:
                # 更新 center
                self._update_center(teacher_out)
                
                # Teacher 输出：centering + normalize
                teacher_out = teacher_out - self.center
                teacher_out = F.normalize(teacher_out, dim=1)
                
                # Softmax with temperature
                student_logits = student_out / self.temperature
                teacher_logits = teacher_out / teacher_temp
                
                # 交叉熵损失（KL divergence）
                loss = -torch.sum(
                    F.softmax(teacher_logits, dim=1) * F.log_softmax(student_logits, dim=1),
                    dim=1
                ).mean()
                
                total_loss += loss
                num_losses += 1
        
        avg_loss = total_loss / max(1, num_losses)
        
        return avg_loss, {"loss": avg_loss.item()}
    
    def update_ema(self):
        """更新 teacher 网络"""
        self._update_teacher()
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch（用于 warmup）"""
        self.current_epoch = epoch


class IBOT(BaseSSLMethod):
    """
    iBOT: Image BERT Pre-Training (满血版本)
    - Loss: CLS 对齐 + patch token 对齐（token-level 预测，仅 ViT）
    - 机制: teacher 网络、mask 生成器、token-level 对齐、multi-crop
    - Backbone: ResNet-50 或 ViT-B/16（token-level 对齐仅在 ViT 时启用）
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        config: Dict[str, Any]
    ):
        super().__init__(backbone, feat_dim, config)
        
        # iBOT 参数
        self.momentum = config.get("momentum", 0.996)
        self.temperature = config.get("temperature", 0.07)
        self.temperature_teacher = config.get("temperature_teacher", 0.04)
        self.center_momentum = config.get("center_momentum", 0.9)
        self.mask_ratio = config.get("mask_ratio", 0.3)  # 30% mask ratio
        
        # 创建 teacher 网络
        self.teacher_backbone = self._build_teacher_encoder()
        self.teacher_head = self._build_teacher_head()
        self._init_teacher()
        
        # Centering
        self.register_buffer("center", torch.zeros(1, self.config.get("proj_output_dim", 8192)))
        self.current_epoch = 0
    
    def _build_teacher_encoder(self) -> nn.Module:
        """构建 teacher encoder"""
        import copy
        teacher = copy.deepcopy(self.backbone)
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher
    
    def _build_teacher_head(self) -> nn.Module:
        """构建 teacher head"""
        import copy
        teacher_head = copy.deepcopy(self.head)
        for param in teacher_head.parameters():
            param.requires_grad = False
        return teacher_head
    
    def _init_teacher(self):
        """初始化 teacher"""
        for param_q, param_k in zip(self.backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.head.parameters(), self.teacher_head.parameters()):
            param_k.data.copy_(param_q.data)
    
    def build_head(self) -> nn.Module:
        """构建 iBOT head（MLP projector，支持 token-level）"""
        proj_hidden_dim = self.config.get("proj_hidden_dim", 2048)
        proj_output_dim = self.config.get("proj_output_dim", 8192)
        
        projector = nn.Sequential(
            nn.Linear(self.feat_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.LayerNorm(proj_output_dim, eps=1e-6)
        )
        return projector
    
    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        """
        iBOT 前向传播：支持返回所有 tokens（仅 ViT）
        
        Args:
            x: [B, 3, H, W] 输入图像
            return_all_tokens: 是否返回所有 tokens（用于 token-level 对齐，仅 ViT 支持）
        
        Returns:
            features: [B, feat_dim] 或 [B, num_tokens, feat_dim]
        """
        h = self.backbone(x)
        
        # 判断是 ViT 还是 ResNet
        is_vit = len(h.shape) == 3  # ViT: [B, num_patches+1, feat_dim], ResNet: [B, feat_dim]
        
        if self.head is not None:
            if return_all_tokens and is_vit:
                # 对所有 tokens 应用 head（仅 ViT）
                B, N, D = h.shape
                h_flat = h.view(B * N, D)
                out_flat = self.head(h_flat)
                out = out_flat.view(B, N, -1)
                return out
            else:
                # 只对 CLS token（ViT）或全局特征（ResNet）应用 head
                if is_vit:
                    cls_token = h[:, 0]  # [B, feat_dim] CLS token
                else:
                    cls_token = h  # [B, feat_dim] ResNet 输出
                return self.head(cls_token)
        return h
    
    def _generate_mask(self, batch_size: int, num_patches: int, device: torch.device) -> torch.Tensor:
        """
        生成随机 mask（用于 patch token 对齐）
        
        Args:
            batch_size: batch 大小
            num_patches: patch 数量（不包括 CLS token）
            device: 设备
        
        Returns:
            mask: [B, num_patches] bool tensor，True 表示 masked
        """
        num_masked = int(num_patches * self.mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            # 随机选择要 mask 的 patches
            idx = torch.randperm(num_patches, device=device)[:num_masked]
            mask[i, idx] = True
        
        return mask
    
    @torch.no_grad()
    def _update_teacher(self):
        """EMA 更新 teacher"""
        for param_q, param_k in zip(self.backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.head.parameters(), self.teacher_head.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        """更新 centering"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def compute_loss(
        self,
        views: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 iBOT 损失（CLS 对齐 + patch token 对齐）
        
        Args:
            views: [B, num_views, 3, H, W] 多个视图（2 global + N local）
        
        Returns:
            loss: 标量损失值
            loss_dict: 损失分解字典
        """
        batch_size = views.size(0)
        num_global = 2
        
        # 分离 global 和 local views
        global_views = views[:, :num_global]  # [B, 2, 3, H, W]
        local_views = views[:, num_global:] if views.size(1) > num_global else None
        
        total_loss = 0
        num_losses = 0
        
        # 1. CLS token 对齐（类似 DINO）
        for i in range(num_global):
            x = global_views[:, i]  # [B, 3, H, W]
            
            # Student CLS
            student_cls = self.forward(x, return_all_tokens=False)  # [B, proj_dim]
            student_cls = F.normalize(student_cls, dim=1)
            
            # Teacher CLS
            with torch.no_grad():
                h_teacher = self.teacher_backbone(x)
                if len(h_teacher.shape) == 3:
                    h_teacher = h_teacher[:, 0]  # CLS token
                teacher_cls = self.teacher_head(h_teacher)  # [B, proj_dim]
                self._update_center(teacher_cls)
                teacher_cls = teacher_cls - self.center
                teacher_cls = F.normalize(teacher_cls, dim=1)
            
            # CLS 对齐损失
            student_logits = student_cls / self.temperature
            teacher_logits = teacher_cls / self.temperature_teacher
            
            loss_cls = -torch.sum(
                F.softmax(teacher_logits, dim=1) * F.log_softmax(student_logits, dim=1),
                dim=1
            ).mean()
            
            total_loss += loss_cls
            num_losses += 1
        
        # 2. Patch token 对齐（只在 ViT 和 global views 上进行）
        # 检查是否是 ViT（通过检查第一个 view 的输出维度）
        with torch.no_grad():
            test_h = self.backbone(global_views[:, 0])
            is_vit = len(test_h.shape) == 3  # ViT: 3D, ResNet: 2D
        
        if is_vit:  # 只在 ViT 时进行 patch token 对齐
            for i in range(num_global):
                x = global_views[:, i]  # [B, 3, H, W]
                
                # Student 所有 tokens
                student_tokens = self.forward(x, return_all_tokens=True)  # [B, num_tokens, proj_dim]
                num_tokens = student_tokens.size(1)
                num_patches = num_tokens - 1  # 不包括 CLS token
                
                # 生成 mask
                mask = self._generate_mask(batch_size, num_patches, student_tokens.device)  # [B, num_patches]
                
                # Teacher 所有 tokens
                with torch.no_grad():
                    h_teacher = self.teacher_backbone(x)  # [B, num_tokens, feat_dim]
                    # 对所有 tokens 应用 head
                    B, N, D = h_teacher.shape
                    h_flat = h_teacher.view(B * N, D)
                    teacher_tokens_flat = self.teacher_head(h_flat)
                    teacher_tokens = teacher_tokens_flat.view(B, N, -1)  # [B, num_tokens, proj_dim]
                    
                    # 更新 center（使用所有 tokens）
                    teacher_tokens_flat_for_center = teacher_tokens.view(B * N, -1)
                    self._update_center(teacher_tokens_flat_for_center)
                    teacher_tokens = teacher_tokens - self.center.unsqueeze(0)
                
                # 只对齐 masked patches（不包括 CLS token）
                student_patches = student_tokens[:, 1:]  # [B, num_patches, proj_dim]
                teacher_patches = teacher_tokens[:, 1:]  # [B, num_patches, proj_dim]
                
                # 选择 masked patches
                student_masked = student_patches[mask]  # [num_masked, proj_dim]
                teacher_masked = teacher_patches[mask]  # [num_masked, proj_dim]
                
                if student_masked.size(0) > 0:
                    student_masked = F.normalize(student_masked, dim=1)
                    teacher_masked = F.normalize(teacher_masked, dim=1)
                    
                    # Patch token 对齐损失
                    student_logits = student_masked / self.temperature
                    teacher_logits = teacher_masked / self.temperature_teacher
                    
                    loss_patch = -torch.sum(
                        F.softmax(teacher_logits, dim=1) * F.log_softmax(student_logits, dim=1),
                        dim=1
                    ).mean()
                    
                    total_loss += loss_patch
                    num_losses += 1
        
        avg_loss = total_loss / max(1, num_losses)
        
        return avg_loss, {"loss": avg_loss.item()}
    
    def update_ema(self):
        """更新 teacher 网络"""
        self._update_teacher()
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch"""
        self.current_epoch = epoch


class VICReg(BaseSSLMethod):
    """
    VICReg: Variance-Invariance-Covariance Regularization
    - Loss: variance / invariance / covariance 三项正则
    """
    
    def build_head(self) -> nn.Module:
        # TODO: 实现 VICReg 的 head
        raise NotImplementedError("VICReg 尚未实现")
    
    def compute_loss(self, views: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        # TODO: 实现 VICReg 的损失
        raise NotImplementedError("VICReg 尚未实现")


class MAE(BaseSSLMethod):
    """
    MAE: Masked Autoencoders
    - Loss: 重建像素/patch 的 L2 损失
    - 机制: encoder + decoder、random mask 生成器
    - Backbone: 仅支持 ViT-B/16 或 ViT-S/16（需要 patch 结构）
    - 注意: MAE 需要 patch-based 架构，不支持 ResNet
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        config: Dict[str, Any]
    ):
        super().__init__(backbone, feat_dim, config)
        
        # MAE 需要 ViT（patch-based 架构）
        # 检查 backbone 输出是否是 3D（ViT 特征）
        # 注意：这里只是警告，实际会在 forward 时检查
        
        # MAE 参数
        self.mask_ratio = config.get("mask_ratio", 0.75)  # 75% mask ratio
        self.patch_size = config.get("patch_size", 16)  # ViT-B/16 的 patch size
        self.img_size = config.get("img_size", 224)
        self.decoder_dim = config.get("decoder_dim", 512)
        self.decoder_depth = config.get("decoder_depth", 8)
        self.decoder_num_heads = config.get("decoder_num_heads", 16)
        
        # 计算 patch 数量
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        # Decoder（在 build_head 中构建）
        # 注意：MAE 的 head 是 decoder
        self.decoder_embed = None
        self.decoder_pos_embed = None
    
    def build_head(self) -> nn.Module:
        """
        构建 MAE decoder
        
        MAE 的结构：
        - Encoder: backbone (ViT)
        - Decoder: 轻量级 Transformer
        """
        # Decoder embedding（将 encoder 输出映射到 decoder 维度）
        self.decoder_embed = nn.Linear(self.feat_dim, self.decoder_dim)
        
        # Decoder positional embedding
        num_patches = self.num_patches
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.decoder_dim)  # +1 for CLS token
        )
        
        # 简单的 Transformer block（用于 decoder）
        class TransformerBlock(nn.Module):
            def __init__(self, dim, num_heads, mlp_dim):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
                self.norm2 = nn.LayerNorm(dim)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim)
                )
            
            def forward(self, x):
                x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                x = x + self.mlp(self.norm2(x))
                return x
        
        # Decoder blocks
        decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.decoder_dim,
                num_heads=self.decoder_num_heads,
                mlp_dim=self.decoder_dim * 4
            )
            for _ in range(self.decoder_depth)
        ])
        
        # Decoder norm
        decoder_norm = nn.LayerNorm(self.decoder_dim)
        
        # 重建头：将 decoder 输出映射回 patch 维度
        decoder_pred = nn.Linear(
            self.decoder_dim,
            self.patch_size * self.patch_size * 3,  # RGB patches
            bias=True
        )
        
        # 组合成完整的 decoder
        class MAEDecoder(nn.Module):
            def __init__(self, embed, pos_embed, blocks, norm, pred):
                super().__init__()
                self.embed = embed
                self.pos_embed = pos_embed
                self.blocks = blocks
                self.norm = norm
                self.pred = pred
            
            def forward(self, x, mask, ids_restore=None):
                """
                Args:
                    x: [B, num_visible, feat_dim] encoder 输出（只包含 visible patches）
                    mask: [B, num_patches] bool tensor，True 表示 masked
                    ids_restore: [B, num_patches] 恢复索引（可选）
                
                Returns:
                    pred: [B, num_patches, patch_size^2 * 3] 重建的 patches（已恢复顺序）
                """
                B = x.size(0)
                num_visible = x.size(1)
                num_patches = mask.size(1)
                
                # Embed
                x = self.embed(x)  # [B, num_visible, decoder_dim]
                
                # 添加 mask tokens
                num_masked = num_patches - num_visible
                mask_tokens = torch.zeros(B, num_masked, self.embed.out_features, device=x.device)
                x = torch.cat([x, mask_tokens], dim=1)  # [B, num_patches, decoder_dim]
                
                # 如果提供了 ids_restore，需要 unshuffle tokens
                if ids_restore is not None:
                    # 恢复原始顺序
                    x_unshuffled = torch.zeros_like(x)
                    for i in range(B):
                        x_unshuffled[i] = x[i, ids_restore[i]]
                    x = x_unshuffled
                
                # 添加 positional embedding
                x = x + self.pos_embed[:, 1:, :]  # 不包括 CLS token 的 pos embed
                
                # 通过 decoder blocks
                for block in self.blocks:
                    x = block(x)
                
                x = self.norm(x)
                
                # 预测 patches
                pred = self.pred(x)  # [B, num_patches, patch_size^2 * 3]
                
                return pred
        
        decoder = MAEDecoder(
            self.decoder_embed,
            self.decoder_pos_embed,
            decoder_blocks,
            decoder_norm,
            decoder_pred
        )
        
        return decoder
    
    def _generate_mask(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成随机 mask
        
        Args:
            batch_size: batch 大小
            device: 设备
        
        Returns:
            mask: [B, num_patches] bool tensor，True 表示 masked
            ids_shuffle: [B, num_patches] 打乱后的 patch 索引
            ids_restore: [B, num_patches] 恢复原始顺序的索引
        """
        num_patches = self.num_patches
        num_masked = int(num_patches * self.mask_ratio)
        num_visible = num_patches - num_masked
        
        # 随机打乱 patch 索引
        ids_shuffle = torch.rand(batch_size, num_patches, device=device).argsort(dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 前 num_visible 个是 visible，后面是 masked
        mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
        mask[:, :num_visible] = False
        
        return mask, ids_shuffle, ids_restore
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, ids_shuffle: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAE 前向传播
        
        Args:
            x: [B, 3, H, W] 输入图像
            mask: [B, num_patches] mask（如果为 None，会生成）
            ids_shuffle: [B, num_patches] shuffle 索引（如果为 None，会生成）
        
        Returns:
            pred: [B, num_patches, patch_size^2 * 3] 重建的 patches
            mask: [B, num_patches] mask
            ids_restore: [B, num_patches] 恢复索引
        """
        B = x.size(0)
        device = x.device
        
        # 生成 mask（如果没有提供）
        if mask is None or ids_shuffle is None:
            mask, ids_shuffle, ids_restore = self._generate_mask(B, device)
        else:
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Encoder 前向传播
        h = self.backbone(x)
        
        # MAE 需要 ViT（3D 输出）
        if len(h.shape) != 3:
            raise ValueError(
                f"MAE 需要 ViT backbone（输出应为 3D），但当前 backbone 输出形状为 {h.shape}。"
                "请使用 --backbone_type vit_b_16"
            )
        
        # ViT 输出: [B, num_patches + 1, feat_dim]
        # 移除 CLS token，只保留 patch tokens
        patch_tokens = h[:, 1:]  # [B, num_patches, feat_dim]
        
        # 根据 ids_shuffle 选择 visible tokens
        # 简化实现：假设每个 batch 的 visible 数量相同
        num_visible = int(self.num_patches * (1 - self.mask_ratio))
        
        # 使用 ids_shuffle 的前 num_visible 个作为 visible tokens
        visible_tokens = []
        for i in range(B):
            visible_idx = ids_shuffle[i, :num_visible]  # 前 num_visible 个是 visible
            visible_tokens_i = patch_tokens[i, visible_idx]  # [num_visible, feat_dim]
            visible_tokens.append(visible_tokens_i)
        
        visible_tokens = torch.stack(visible_tokens, dim=0)  # [B, num_visible, feat_dim]
        
        # Decoder 重建
        pred = self.head(visible_tokens, mask, ids_restore)  # [B, num_patches, patch_size^2 * 3]
        
        return pred, mask, ids_restore
    
    def compute_loss(
        self,
        views: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 MAE 重建损失
        
        Args:
            views: [B, 1, 3, H, W] MAE 只需要 1 个 view
        
        Returns:
            loss: 标量损失值
            loss_dict: 损失分解字典
        """
        # MAE 只需要 1 个 view
        x = views[:, 0]  # [B, 3, H, W]
        
        # 生成 mask
        B = x.size(0)
        device = x.device
        mask, ids_shuffle, ids_restore = self._generate_mask(B, device)
        
        # 前向传播
        pred, mask, ids_restore = self.forward(x, mask, ids_shuffle)
        
        # 将图像转换为 patches
        # x: [B, 3, H, W] -> [B, num_patches, patch_size^2 * 3]
        patches = self._image_to_patches(x)  # [B, num_patches, patch_size^2 * 3]
        
        # 根据 ids_restore 恢复 patches 的顺序
        patches_restored = torch.zeros_like(patches)
        for i in range(B):
            patches_restored[i] = patches[i, ids_restore[i]]
        
        # 只计算 masked patches 的损失
        # pred 已经是恢复顺序后的预测
        pred_masked = pred[mask]  # [num_masked, patch_size^2 * 3]
        target_masked = patches_restored[mask]  # [num_masked, patch_size^2 * 3]
        
        # L2 损失
        loss = F.mse_loss(pred_masked, target_masked)
        
        return loss, {"loss": loss.item()}
    
    def _image_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        将图像转换为 patches
        
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            patches: [B, num_patches, patch_size^2 * 3]
        """
        B, C, H, W = x.shape
        P = self.patch_size
        
        # 确保图像尺寸是 patch_size 的倍数
        assert H % P == 0 and W % P == 0
        
        # 分割成 patches
        patches = x.unfold(2, P, P).unfold(3, P, P)  # [B, C, H//P, W//P, P, P]
        patches = patches.contiguous().view(B, C, -1, P, P)  # [B, C, num_patches, P, P]
        patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, P, P]
        patches = patches.contiguous().view(B, -1, C * P * P)  # [B, num_patches, patch_size^2 * 3]
        
        return patches
    
    def get_views(self, images: torch.Tensor) -> torch.Tensor:
        """MAE 只需要 1 个 view"""
        # 如果输入是 [B, num_views, 3, H, W]，只取第一个
        if len(images.shape) == 5:
            return images[:, 0:1]  # [B, 1, 3, H, W]
        return images.unsqueeze(1)  # [B, 1, 3, H, W]


# ============================================================
# 方法工厂函数
# ============================================================

def build_method(
    method_name: str,
    backbone_type: str = "resnet50",
    pretrained_backbone: bool = False,
    config: Dict[str, Any] = None
) -> BaseSSLMethod:
    """
    构建自监督学习方法
    
    Args:
        method_name: 方法名称（"simclr", "moco", "byol", "dino", "ibot", "vicreg", "mae"）
        backbone_type: backbone 类型（"resnet50" 或 "vit_b_16"）
        pretrained_backbone: 是否使用预训练 backbone
        config: 方法特定的配置字典
    
    Returns:
        method: 方法实例
    """
    if config is None:
        config = {}
    
    # 构建 backbone
    backbone, feat_dim = build_backbone(backbone_type, pretrained_backbone)
    
    # 创建方法实例
    method_classes = {
        "simclr": SimCLR,
        "moco": MoCo,
        "byol": BYOL,
        "dino": DINO,
        "ibot": IBOT,
        "vicreg": VICReg,
        "mae": MAE,
    }
    
    if method_name.lower() not in method_classes:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(method_classes.keys())}")
    
    method_class = method_classes[method_name.lower()]
    method = method_class(backbone, feat_dim, config)
    
    return method
