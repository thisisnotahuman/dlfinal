"""
通用训练循环 - 完全可以共用的部分
==================================================
统一的训练循环外壳，每个方法只需要实现 compute_loss()
"""

import os
import argparse
from datetime import datetime

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from load_data import load_dino_data
from model import build_method
from utils import (
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    count_parameters
)


# ============================================================
# 通用训练循环
# ============================================================

def train_ssl(
    method,
    train_loader,
    device,
    optimizer,
    scheduler,
    epochs,
    save_dir,
    use_amp=True,
    save_freq=1,
    log_freq=100,
    two_view_aug=None  # 增强函数
):
    """
    通用自监督学习训练循环
    
    Args:
        method: 自监督学习方法实例（继承 BaseSSLMethod）
        train_loader: 训练数据加载器
        device: 设备
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数
        save_dir: 保存目录
        use_amp: 是否使用自动混合精度
        save_freq: 保存频率（每 N 个 epoch）
        log_freq: 日志频率（每 N 个 step）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    scaler = GradScaler() if use_amp else None
    
    print("\n🚀 开始训练...")
    print(f"   方法: {method.__class__.__name__}")
    print(f"   参数量: {count_parameters(method):,}")
    print(f"   设备: {device}")
    print(f"   AMP: {use_amp}")
    print()
    
    global_step = 0
    best_loss = float("inf")
    
    # Epoch 循环
    for epoch in range(1, epochs + 1):
        method.train()
        epoch_loss = 0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        
        for batch in pbar:
            # batch 是 [B, 3, H, W] CPU tensor，需要移到 GPU 并进行增强
            batch = batch.to(device, non_blocking=True)  # [B, 3, H, W] GPU
            
            # 应用增强（生成 views）- 在主进程的 GPU 上进行
            if two_view_aug is not None:
                batch = two_view_aug(batch)  # [B, 2, 3, H, W] GPU
            
            optimizer.zero_grad()
            
            # 前向传播和损失计算
            if use_amp:
                with autocast():
                    views = method.get_views(batch)
                    loss, loss_dict = method.compute_loss(views)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                views = method.get_views(batch)
                loss, loss_dict = method.compute_loss(views)
                loss.backward()
                optimizer.step()
            
            # 更新 EMA（如果有 teacher 网络）
            method.update_ema()
            
            # 更新学习率（某些方法可能需要在 step 级别更新）
            if hasattr(scheduler, 'step') and callable(getattr(scheduler, 'step', None)):
                # 检查是否是 epoch 级别的 scheduler
                if not hasattr(scheduler, 'current_epoch'):
                    # 如果是 step 级别的，在这里更新
                    pass  # 暂时不在这里更新，在 epoch 结束后更新
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # 日志
            if global_step % log_freq == 0:
                pbar.set_postfix({**loss_dict, "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        # Epoch 结束：更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 设置 epoch（用于 DINO/iBOT 的 warmup）
        if hasattr(method, 'set_epoch'):
            method.set_epoch(epoch)
        
        avg_loss = epoch_loss / max(1, num_batches)
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        
        print(f"\n📌 Epoch {epoch}/{epochs}:")
        print(f"   avg_loss = {avg_loss:.4f}")
        print(f"   lr = {current_lr:.3e}")
        
        # 保存 checkpoint
        # 保存 checkpoint
        if epoch % save_freq == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": method.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "global_step": global_step,
            }
            
            if scheduler is not None:
                ckpt["scheduler_state_dict"] = (
                    scheduler.scheduler.state_dict() 
                    if hasattr(scheduler, 'scheduler') 
                    else scheduler.state_dict()
                )
            
            save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pth")
            # ⬅️ 直接保存 ckpt，不再传 epoch, method, optimizer, scheduler
            torch.save(ckpt, save_path)
            print(f"💾 保存模型到 {save_path}")

        # 保存 best 模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best.pth")
            ckpt = {
                "epoch": epoch,
                "model_state_dict": method.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "global_step": global_step,
            }
            if scheduler is not None:
                ckpt["scheduler_state_dict"] = (
                    scheduler.scheduler.state_dict() 
                    if hasattr(scheduler, 'scheduler') 
                    else scheduler.state_dict()
                )
            torch.save(ckpt, best_path)
            print(f"🏅 更新 Best 模型（loss={best_loss:.4f}）")


# ============================================================
# 主训练函数
# ============================================================

def main_train(args):
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 设备: {device}")
    
    # 加载数据
    train_loader, _, _, two_view_aug = load_dino_data(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_sample=args.train_sample,
        eval_samples=None,
        strength=args.aug_strength,
    )
    
    # 将 two_view_aug 存储为全局变量或传递给训练函数
    # 为了简化，我们修改训练循环来使用它
    
    # 构建方法配置
    method_config = {
        "proj_hidden_dim": args.proj_hidden_dim,
        "proj_output_dim": args.proj_output_dim,
        "temperature": args.temperature,
        # 可以添加其他方法特定的配置
    }
    
    # 构建方法
    method = build_method(
        method_name=args.method,
        backbone_type=args.backbone_type,
        pretrained_backbone=args.pretrained_backbone,
        config=method_config
    ).to(device)
    
    # 构建优化器和调度器
    optimizer = build_optimizer(
        method,
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=args.scheduler_type,
        T_max=args.epochs,
        warmup_epochs=args.warmup_epochs
    )
    
    # 训练
    train_ssl(
        method=method,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        save_dir=args.save_dir,
        use_amp=args.use_amp,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        two_view_aug=two_view_aug  # 传递增强函数
    )


# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("通用自监督学习训练框架")
    
    # 方法选择
    parser.add_argument("--method", type=str, default="simclr",
                       choices=["simclr", "moco", "byol", "dino", "ibot", "vicreg", "mae"],
                       help="自监督学习方法")
    
    # 数据
    parser.add_argument("--dataset_type", type=str, default="huggingface")
    parser.add_argument("--dataset_name", type=str, default="tsbpp/fall2025_deeplearning")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_sample", type=int, default=None,
                       help="训练集子集大小，例如 50000")
    parser.add_argument("--aug_strength", type=str, default="strong",
                       choices=["strong", "weak"])
    
    # 模型
    parser.add_argument("--backbone_type", type=str, default="resnet50",
                       choices=["resnet50", "vit_b_16"])
    parser.add_argument("--pretrained_backbone", action="store_true",
                       help="是否使用预训练 backbone")
    
    # 方法特定参数（SimCLR）
    parser.add_argument("--proj_hidden_dim", type=int, default=2048)
    parser.add_argument("--proj_output_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)
    
    # 训练
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer_type", type=str, default="adamw",
                       choices=["adamw", "sgd"])
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                       choices=["cosine", "step"])
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="使用自动混合精度")
    
    # 保存和日志
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_freq", type=int, default=1,
                       help="每 N 个 epoch 保存一次")
    parser.add_argument("--log_freq", type=int, default=100,
                       help="每 N 个 step 记录一次日志")
    
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    torch.backends.cudnn.benchmark = True
    print("=" * 60)
    print("通用自监督学习训练框架")
    print(f"启动时间: {datetime.now()}")
    print("=" * 60)
    
    args = parse_args()
    main_train(args)
    
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
