"""
评估 Pipeline - 完全可以共用的部分
==================================================
从 frozen encoder 抽 feature → 建 k-NN feature bank → 在 eval 上算 accuracy
或 linear probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
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
    use_cls_token: bool = False,
    disable_tqdm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 frozen encoder 提取特征
    
    Args:
        model: 编码器模型（backbone 或 backbone + head 的一部分）
        dataloader: 数据加载器
        device: 设备
        use_cls_token: 是否使用 CLS token（ViT）
        disable_tqdm: 是否禁用 tqdm 进度条
    
    Returns:
        features: [N, feat_dim] 特征矩阵
        labels: [N] 标签（如果有）
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        iterator = dataloader if disable_tqdm else tqdm(dataloader, desc="提取特征")
        for batch in iterator:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, batch_labels = batch
                # ✅ 性能优化：在 GPU 上累积标签，最后一次性移到 CPU
                labels.append(batch_labels)  # 保持在 GPU 上
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
            # 对于 ViT（3D 输出），总是取 CLS token（第 0 个 token）
            # use_cls_token 参数主要用于控制是否显式指定使用 CLS token
            if len(feat.shape) == 3:
                # [B, num_tokens, feat_dim] -> [B, feat_dim] (取 CLS token)
                # ViT 输出: [B, num_patches+1, feat_dim]，第 0 个是 CLS token
                feat = feat[:, 0]
            elif len(feat.shape) != 2:
                # 如果不是 2D 或 3D，报错
                raise ValueError(f"Unexpected feature shape: {feat.shape}, expected 2D [B, D] or 3D [B, N, D]")
            
            # 归一化
            feat = F.normalize(feat, dim=1)
            
            # 检查 NaN 和 Inf
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                print(f"⚠️  Warning: NaN/Inf detected in features, replacing with zeros")
                feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
                # 重新归一化
                feat = F.normalize(feat, dim=1)
            
            # ✅ 性能优化：在 GPU 上累积特征，最后一次性移到 CPU
            features.append(feat)  # 保持在 GPU 上
    
    # ✅ 性能优化：在 GPU 上拼接，然后一次性移到 CPU
    features = torch.cat(features, dim=0).cpu().numpy()
    
    # 最终检查 NaN
    if np.isnan(features).any() or np.isinf(features).any():
        print(f"⚠️  Warning: NaN/Inf in final features, replacing with zeros")
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        # 重新归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # 避免除以0
        features = features / norms
    
    # ✅ 性能优化：如果有标签，在 GPU 上拼接后一次性移到 CPU
    if labels and len(labels) > 0:
        if isinstance(labels[0], torch.Tensor):
            # 标签是 GPU tensor，在 GPU 上拼接
            labels = torch.cat(labels, dim=0).cpu().numpy()
        else:
            # 标签已经是 numpy，直接转换
            labels = np.array(labels)
    else:
        labels = None
    
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
    linear_probe_C: float = 1.0,
    disable_tqdm: bool = False
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
        disable_tqdm: 是否禁用 tqdm 进度条
    
    Returns:
        results: 评估结果字典
    """
    print("=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    # 提取特征
    print("\n1. 提取训练集特征...")
    train_features, train_labels = extract_features(
        model, train_loader, device, use_cls_token, disable_tqdm
    )
    print(f"   训练集特征形状: {train_features.shape}")
    if train_labels is not None:
        print(f"   训练集标签范围: [{train_labels.min()}, {train_labels.max()}], 类别数: {len(np.unique(train_labels))}")
        print(f"   训练集标签分布: {np.bincount(train_labels.astype(int))[:10]}... (前10个类别)")
    
    print("\n2. 提取验证集特征...")
    val_features, val_labels = extract_features(
        model, val_loader, device, use_cls_token, disable_tqdm
    )
    print(f"   验证集特征形状: {val_features.shape}")
    if val_labels is not None:
        print(f"   验证集标签范围: [{val_labels.min()}, {val_labels.max()}], 类别数: {len(np.unique(val_labels))}")
        print(f"   验证集标签分布: {np.bincount(val_labels.astype(int))[:10]}... (前10个类别)")
    
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


# ============================================================
# 5. CUB 数据集加载（用于评估）
# ============================================================

class CUBDataset(Dataset):
    """CUB-200-2011 数据集"""
    
    def __init__(self, image_dir, labels_csv=None, transform=None):
        """
        Args:
            image_dir: 图片文件夹路径
            labels_csv: 标签CSV文件（train/val有，test没有）
            transform: 图像变换
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # 加载标签（如果有）
        if labels_csv is not None:
            self.labels_df = pd.read_csv(labels_csv)
            self.has_labels = True
        else:
            # Test set: 只有图片列表
            self.labels_df = pd.DataFrame({
                'filename': [f.name for f in self.image_dir.glob('*.jpg')]
            })
            self.has_labels = False
        
        print(f"✔ Loaded {len(self.labels_df)} images from {image_dir}")
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # 加载图片
        img_path = self.image_dir / row['filename']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        if self.has_labels:
            label = row['class_id']
            return img, label
        else:
            return img


def load_cub_data(data_dir, img_size=96, batch_size=256, num_workers=4):
    """
    加载 CUB-200-2011 评估数据
    
    Args:
        data_dir: kaggle_data/ 文件夹路径
        img_size: 图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        train_loader: 训练集（用于 k-NN feature bank）
        val_loader: 验证集（用于评估）
    """
    data_dir = Path(data_dir)
    
    # 图像变换（不做增强）
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CUBDataset(
        image_dir=data_dir / 'train',
        labels_csv=data_dir / 'train_labels.csv',
        transform=transform
    )
    
    val_dataset = CUBDataset(
        image_dir=data_dir / 'val',
        labels_csv=data_dir / 'val_labels.csv',
        transform=transform
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================
# 6. CUB 评估函数（模块化，可在训练中使用）
# ============================================================

def evaluate_on_cub(
    method: nn.Module,
    cub_data_dir: str,
    device: torch.device,
    img_size: int = 96,
    batch_size: int = 256,
    num_workers: int = 4,
    eval_method: str = "knn",
    use_cls_token: bool = False,
    knn_k: int = 20,
    linear_probe_C: float = 1.0,
    verbose: bool = True,
    disable_tqdm: bool = False
) -> Dict[str, float]:
    """
    在 CUB-200-2011 上评估模型（模块化函数，可在训练循环中使用）
    
    Args:
        method: 自监督学习方法实例（需要实现 get_encoder() 方法）
        cub_data_dir: CUB 数据文件夹路径（包含 train/val/test）
        device: 设备
        img_size: 图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载线程数
        eval_method: 评估方法，"knn" 或 "linear_probe"
        use_cls_token: 是否使用 CLS token（仅 ViT）
        knn_k: k-NN 的 k 值
        linear_probe_C: Linear Probe 的正则化强度
        verbose: 是否打印详细信息
        disable_tqdm: 是否禁用 tqdm 进度条
    
    Returns:
        results: 评估结果字典，包含 accuracy 和 eval_method
    """
    if verbose:
        print("\n" + "="*60)
        print("📊 在 CUB-200-2011 上评估模型...")
        print("="*60)
    
    # 加载 CUB 数据
    if verbose:
        print("加载 CUB-200-2011 数据集...")
    train_loader, val_loader = load_cub_data(
        cub_data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # 获取编码器（冻结）
    encoder = method.get_encoder()
    encoder.eval()
    
    # 评估
    results = evaluate_model(
        model=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        eval_method=eval_method,
        use_cls_token=use_cls_token,
        knn_k=knn_k,
        linear_probe_C=linear_probe_C,
        disable_tqdm=disable_tqdm
    )
    
    if verbose:
        print(f"\n🎯 CUB-200-2011 {eval_method} 准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("="*60)
    
    return results

