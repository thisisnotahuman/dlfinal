"""
在 CUB-200-2011 上评估预训练模型
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import argparse

from model import build_method
from eval import extract_features, knn_eval, linear_probe_eval


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


def main():
    parser = argparse.ArgumentParser("在 CUB-200-2011 上评估预训练模型")
    
    # 模型参数
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="预训练模型的 checkpoint 路径")
    parser.add_argument("--method", type=str, default="simclr")
    parser.add_argument("--backbone_type", type=str, default="resnet50")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="CUB 数据文件夹路径（包含 train/val/test）")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 评估参数
    parser.add_argument("--eval_method", type=str, default="knn",
                       choices=["knn", "linear_probe"])
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--linear_probe_C", type=float, default=1.0)
    parser.add_argument("--use_cls_token", action="store_true",
                       help="是否使用 CLS token（仅 ViT）")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 设备: {device}")
    
    # 加载评估数据
    print("\n" + "="*60)
    print("📊 加载 CUB-200-2011 数据集...")
    print("="*60)
    
    train_loader, val_loader = load_cub_data(
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.num_workers
    )
    
    # 构建方法
    method_config = {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 128,
        "temperature": 0.5,
    }
    
    method = build_method(
        method_name=args.method,
        backbone_type=args.backbone_type,
        pretrained_backbone=False,
        config=method_config
    ).to(device)
    
    # 加载 checkpoint
    print(f"\n📥 加载 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    method.load_state_dict(checkpoint["model_state_dict"])
    print(f"✔ 已加载 epoch {checkpoint.get('epoch', '?')}, loss={checkpoint.get('avg_loss', '?'):.4f}")
    
    # 获取编码器（冻结）
    encoder = method.get_encoder()
    encoder.eval()
    
    # 提取特征
    print("\n" + "="*60)
    print("🔍 提取特征...")
    print("="*60)
    
    print("\n1. 提取训练集特征（用于 feature bank）...")
    train_features, train_labels = extract_features(
        encoder, train_loader, device, args.use_cls_token
    )
    print(f"   训练集特征: {train_features.shape}")
    
    print("\n2. 提取验证集特征...")
    val_features, val_labels = extract_features(
        encoder, val_loader, device, args.use_cls_token
    )
    print(f"   验证集特征: {val_features.shape}")
    
    # 评估
    print("\n" + "="*60)
    print(f"📈 {args.eval_method.upper()} 评估...")
    print("="*60)
    
    if args.eval_method == "knn":
        accuracy = knn_eval(
            train_features, train_labels,
            val_features, val_labels,
            k=args.knn_k
        )
    elif args.eval_method == "linear_probe":
        accuracy = linear_probe_eval(
            train_features, train_labels,
            val_features, val_labels,
            C=args.linear_probe_C
        )
    
    # 打印结果
    print("\n" + "="*60)
    print("🎯 最终结果")
    print("="*60)
    print(f"方法: {args.eval_method}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"验证集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()