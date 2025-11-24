"""
评估 Pipeline - 完全可以共用的部分
==================================================
从 frozen encoder 抽 feature → 建 k-NN feature bank → 在 eval 上算 accuracy
或 linear probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ============================================================
# 1. 特征提取
# ============================================================

def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_cls_token: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 frozen encoder 提取特征
    
    Args:
        model: 编码器模型（backbone 或 backbone + head 的一部分）
        dataloader: 数据加载器
        device: 设备
        use_cls_token: 是否使用 CLS token（ViT）
    
    Returns:
        features: [N, feat_dim] 特征矩阵
        labels: [N] 标签（如果有）
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, batch_labels = batch
                labels.extend(batch_labels.cpu().numpy())
            else:
                images = batch
                if isinstance(images, (list, tuple)):
                    images = images[0]  # 如果是 views，取第一个
            
            images = images.to(device)
            
            # 提取特征
            if hasattr(model, 'forward_features'):
                # 某些模型可能有 forward_features 方法
                feat = model.forward_features(images)
            else:
                feat = model(images)
            
            # 处理 ViT 的 CLS token
            if use_cls_token and len(feat.shape) == 3:
                # [B, num_tokens, feat_dim] -> [B, feat_dim] (取 CLS token)
                feat = feat[:, 0]
            
            # 归一化
            feat = F.normalize(feat, dim=1)
            
            features.append(feat.cpu())
    
    features = torch.cat(features, dim=0).numpy()
    labels = np.array(labels) if labels else None
    
    return features, labels


# ============================================================
# 2. k-NN 评估
# ============================================================

def knn_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    k: int = 20
) -> float:
    """
    使用 k-NN 在特征空间上评估
    
    Args:
        train_features: [N_train, feat_dim] 训练集特征
        train_labels: [N_train] 训练集标签
        val_features: [N_val, feat_dim] 验证集特征
        val_labels: [N_val] 验证集标签
        k: k-NN 的 k 值
    
    Returns:
        accuracy: 准确率
    """
    print(f"训练 k-NN (k={k})...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)
    
    print("预测...")
    pred_labels = knn.predict(val_features)
    accuracy = accuracy_score(val_labels, pred_labels)
    
    return accuracy


# ============================================================
# 3. Linear Probe 评估
# ============================================================

def linear_probe_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    max_iter: int = 1000,
    C: float = 1.0
) -> float:
    """
    使用 Linear Probe 评估
    
    Args:
        train_features: [N_train, feat_dim] 训练集特征
        train_labels: [N_train] 训练集标签
        val_features: [N_val, feat_dim] 验证集特征
        val_labels: [N_val] 验证集标签
        max_iter: 最大迭代次数
        C: 正则化强度
    
    Returns:
        accuracy: 准确率
    """
    print(f"训练 Linear Probe (C={C})...")
    lr = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver='lbfgs',
        multi_class='multinomial'
    )
    lr.fit(train_features, train_labels)
    
    print("预测...")
    pred_labels = lr.predict(val_features)
    accuracy = accuracy_score(val_labels, pred_labels)
    
    return accuracy


# ============================================================
# 4. 完整评估 Pipeline
# ============================================================

def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_method: str = "knn",  # "knn" 或 "linear_probe"
    use_cls_token: bool = False,
    knn_k: int = 20,
    linear_probe_C: float = 1.0
) -> Dict[str, float]:
    """
    完整的评估 pipeline
    
    Args:
        model: 编码器模型
        train_loader: 训练集数据加载器（用于构建 feature bank）
        val_loader: 验证集数据加载器
        device: 设备
        eval_method: "knn" 或 "linear_probe"
        use_cls_token: 是否使用 CLS token（ViT）
        knn_k: k-NN 的 k 值
        linear_probe_C: Linear Probe 的正则化强度
    
    Returns:
        results: 评估结果字典
    """
    print("=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    # 提取特征
    print("\n1. 提取训练集特征...")
    train_features, train_labels = extract_features(
        model, train_loader, device, use_cls_token
    )
    print(f"   训练集特征形状: {train_features.shape}")
    
    print("\n2. 提取验证集特征...")
    val_features, val_labels = extract_features(
        model, val_loader, device, use_cls_token
    )
    print(f"   验证集特征形状: {val_features.shape}")
    
    # 评估
    print(f"\n3. {eval_method} 评估...")
    if eval_method == "knn":
        accuracy = knn_eval(
            train_features, train_labels,
            val_features, val_labels,
            k=knn_k
        )
    elif eval_method == "linear_probe":
        accuracy = linear_probe_eval(
            train_features, train_labels,
            val_features, val_labels,
            C=linear_probe_C
        )
    else:
        raise ValueError(f"Unknown eval_method: {eval_method}")
    
    results = {
        "accuracy": accuracy,
        "eval_method": eval_method
    }
    
    print(f"\n✅ 评估完成: {eval_method} accuracy = {accuracy:.4f}")
    print("=" * 60)
    
    return results

