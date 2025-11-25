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
import wandb

from load_data import load_dino_data
from model import build_method
from utils import (
    build_optimizer,
    build_scheduler,
    save_checkpoint,  # 目前没用到，但先留着
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
    two_view_aug,
    use_amp=True,
    save_freq=1,
    log_freq=100,
    use_wandb=False,
    wandb_project="ssl-pretraining",
    wandb_name=None,
    early_stop_patience=None,
    early_stop_min_delta=0.0001,
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
        two_view_aug: 数据增强函数（输入一批图像，输出两组增强视图）
        use_amp: 是否使用自动混合精度
        save_freq: 保存频率（每 N 个 epoch）
        log_freq: 日志频率（每 N 个 step）
        use_wandb: 是否使用 wandb 监控
        wandb_project: wandb 项目名
        wandb_name: wandb 运行名称
        early_stop_patience: 早停耐心值（连续多少个epoch没有改善则停止），None表示不使用早停
        early_stop_min_delta: 早停最小改善阈值
    """
    os.makedirs(save_dir, exist_ok=True)

    scaler = GradScaler() if use_amp else None

    print("\n🚀 开始训练...")
    print(f"   方法: {method.__class__.__name__}")
    print(f"   参数量: {count_parameters(method):,}")
    print(f"   设备: {device}")
    print(f"   AMP: {use_amp}")
    if use_wandb:
        print(f"   Wandb: ✅ {wandb_project}/{wandb_name}")
    if early_stop_patience is not None:
        print(f"   早停: ✅ patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    print()

    global_step = 0
    best_loss = float("inf")
    epochs_without_improvement = 0  # 早停计数器

    # Epoch 循环
    for epoch in range(1, epochs + 1):
        method.train()
        epoch_loss = 0.0
        num_batches = 0

        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)

        for batch in pbar:
            # 将 batch 移到设备并做两视图增强
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device, non_blocking=True)
                views = two_view_aug(batch)
            elif isinstance(batch, (list, tuple)):
                batch = [
                    b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b
                    for b in batch
                ]
                views = method.get_views(batch)
            else:
                views = method.get_views(batch)

            optimizer.zero_grad()

            # 前向传播和损失计算
            if use_amp:
                with autocast():
                    loss, loss_dict = method.compute_loss(views)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = method.compute_loss(views)
                loss.backward()
                optimizer.step()

            # 更新 EMA（如果有 teacher 网络）
            method.update_ema()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Step 级别日志
            if global_step % log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({**loss_dict, "lr": f"{current_lr:.2e}"})

                if use_wandb:
                    wandb.log(
                        {
                            "train/loss_step": loss.item(),
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            **{f"train/{k}": v for k, v in loss_dict.items()},
                        },
                        step=global_step,
                    )

        # Epoch 结束：更新学习率
        if scheduler is not None:
            scheduler.step()

        # 设置 epoch（用于 DINO/iBOT 的 warmup 等）
        if hasattr(method, "set_epoch"):
            method.set_epoch(epoch)

        avg_loss = epoch_loss / max(1, num_batches)
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else optimizer.param_groups[0]["lr"]
        )

        print(f"\n📌 Epoch {epoch}/{epochs}:")
        print(f"   avg_loss = {avg_loss:.4f}")
        print(f"   lr = {current_lr:.3e}")

        # Epoch 级别日志
        if use_wandb:
            wandb.log(
                {
                    "train/loss_epoch": avg_loss,
                    "train/lr_epoch": current_lr,
                    "epoch": epoch,
                },
                step=global_step,
            )

        # 保存当前 epoch 的 checkpoint
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
                    if hasattr(scheduler, "scheduler")
                    else scheduler.state_dict()
                )

            save_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pth")
            torch.save(ckpt, save_path)
            print(f"💾 保存模型到 {save_path}")

            if use_wandb:
                wandb.save(save_path)

        # 保存 best 模型 & 早停逻辑
        if avg_loss < best_loss - early_stop_min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0  # 重置早停计数器

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
                    if hasattr(scheduler, "scheduler")
                    else scheduler.state_dict()
                )
            torch.save(ckpt, best_path)
            print(f"🏅 更新 Best 模型（loss={best_loss:.4f}）")

            if use_wandb:
                wandb.run.summary["best_loss"] = best_loss
                wandb.run.summary["best_epoch"] = epoch
                wandb.save(best_path)
        else:
            epochs_without_improvement += 1
            if early_stop_patience is not None:
                print(f"⚠️  Loss 没有改善 ({epochs_without_improvement}/{early_stop_patience})")

        # 早停检查
        if (
            early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print(f"\n🛑 早停触发！连续 {early_stop_patience} 个 epoch 没有改善")
            print(f"   Best loss: {best_loss:.4f} (Epoch {epoch - early_stop_patience})")
            if use_wandb:
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["stopped_epoch"] = epoch
            break


# ============================================================
# 主训练函数
# ============================================================

def main_train(args):
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 设备: {device}")

    # 初始化 wandb
    if args.use_wandb:
        wandb_name = (
            args.wandb_name
            or f"{args.method}_{args.backbone_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config={
                "method": args.method,
                "backbone_type": args.backbone_type,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "optimizer_type": args.optimizer_type,
                "scheduler_type": args.scheduler_type,
                "warmup_epochs": args.warmup_epochs,
                "temperature": args.temperature,
                "proj_hidden_dim": args.proj_hidden_dim,
                "proj_output_dim": args.proj_output_dim,
                "aug_strength": args.aug_strength,
                "train_sample": args.train_sample,
                "dataset_type": args.dataset_type,
                "dataset_root": args.dataset_root,
                "dataset_name": args.dataset_name,
            },
        )

    # 加载数据（这里合并本地 / HF 入口）
    train_loader, _, _, two_view_aug = load_dino_data(
        dataset_type=args.dataset_type,     # "local" 或 "huggingface"
        dataset_root=args.dataset_root,     # 本地时使用
        dataset_name=args.dataset_name,     # HF 时使用
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_sample=args.train_sample,
        strength=args.aug_strength,
    )

    # 构建方法配置
    method_config = {
        "proj_hidden_dim": args.proj_hidden_dim,
        "proj_output_dim": args.proj_output_dim,
        "temperature": args.temperature,
    }

    # 构建方法
    method = build_method(
        method_name=args.method,
        backbone_type=args.backbone_type,
        pretrained_backbone=args.pretrained_backbone,
        config=method_config,
    ).to(device)

    # Wandb watch model（可选）
    if args.use_wandb and args.wandb_watch:
        wandb.watch(method, log="all", log_freq=args.log_freq)

    # 构建优化器和调度器
    optimizer = build_optimizer(
        method,
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        scheduler_type=args.scheduler_type,
        T_max=args.epochs,
        warmup_epochs=args.warmup_epochs,
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
        two_view_aug=two_view_aug,
        use_amp=args.use_amp,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    # 关闭 wandb
    if args.use_wandb:
        wandb.finish()


# ============================================================
# Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("通用自监督学习训练框架")

    # 方法选择
    parser.add_argument(
        "--method",
        type=str,
        default="simclr",
        choices=["simclr", "moco", "byol", "dino", "ibot", "vicreg", "mae"],
        help="自监督学习方法",
    )

    # 数据（本地 / HuggingFace）
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="local",
        choices=["local", "huggingface"],
        help="数据来源：本地文件夹或 HuggingFace 数据集",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="images/train",
        help="本地图片路径，例如 images/train（dataset_type=local 时使用）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace 数据集名（仅当 dataset_type=huggingface 时使用）",
    )
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--train_sample",
        type=int,
        default=None,
        help="训练集子集大小，例如 50000（None 表示用全部）",
    )
    parser.add_argument(
        "--aug_strength",
        type=str,
        default="strong",
        choices=["strong", "weak"],
        help="数据增强强度",
    )

    # 模型
    parser.add_argument(
        "--backbone_type",
        type=str,
        default="resnet50",
        choices=["resnet50", "vit_b_16"],
    )
    parser.add_argument(
        "--pretrained_backbone",
        action="store_true",
        help="是否使用预训练 backbone",
    )

    # 方法特定参数（比如 SimCLR）
    parser.add_argument("--proj_hidden_dim", type=int, default=2048)
    parser.add_argument("--proj_output_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)

    # 训练
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "step"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="使用自动混合精度（不加该参数则为 False）",
    )

    # 保存和日志
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ssl-pretraining")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_watch", action="store_true")

    # 早停
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="早停耐心值，None 表示不使用早停",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0001,
        help="早停最小改善阈值",
    )

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
