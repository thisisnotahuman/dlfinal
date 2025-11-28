"""
通用训练循环 - 完全可以共用的部分
==================================================
统一的训练循环外壳，每个方法只需要实现 compute_loss()
"""

import os
import argparse
import time
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
from eval import evaluate_on_cub


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
    save_freq=None,
    log_freq=100,
    use_wandb=False,
    wandb_project="ssl-pretraining",
    wandb_name=None,
    early_stop_patience=None,
    early_stop_min_delta=0.0001,
    # 恢复训练参数
    start_epoch=1,  # 从哪个 epoch 开始（用于恢复训练）
    start_global_step=0,  # 从哪个 global_step 开始（用于恢复训练）
    start_best_loss=float("inf"),  # 初始 best_loss（用于恢复训练）
    # 评估相关参数
    eval_enabled=False,
    eval_cub_data_dir=None,
    eval_freq=2,  # 每 N 个 epoch 评估一次
    eval_method="knn",
    eval_knn_k=20,
    eval_linear_probe_C=1.0,
    eval_use_cls_token=False,
    eval_batch_size=256,
    eval_num_workers=4,
    img_size=96,  # 图像尺寸（用于评估）
    disable_tqdm=False,  # 是否禁用 tqdm 进度条
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
        eval_enabled: 是否启用评估
        eval_cub_data_dir: CUB 数据文件夹路径（eval_enabled=True 时必需）
        eval_freq: 评估频率（每 N 个 epoch 评估一次）
        eval_method: 评估方法，"knn" 或 "linear_probe"
        eval_knn_k: k-NN 的 k 值
        eval_linear_probe_C: Linear Probe 的正则化强度
        eval_use_cls_token: 是否使用 CLS token（仅 ViT）
        eval_batch_size: 评估时的批次大小
        eval_num_workers: 评估时的数据加载线程数
        img_size: 图像尺寸（用于评估）
        disable_tqdm: 是否禁用 tqdm 进度条
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
    if eval_enabled:
        print(f"   评估: ✅ 每 {eval_freq} 个 epoch 在 CUB-200-2011 上评估 ({eval_method})")
        print(f"         CUB 数据路径: {eval_cub_data_dir}")
    print()

    global_step = start_global_step  # ✅ 恢复训练：从指定 global_step 开始
    best_loss = start_best_loss  # ✅ 恢复训练：从指定 best_loss 开始
    epochs_without_improvement = 0  # 早停计数器

    # Epoch 循环
    for epoch in range(start_epoch, epochs + 1):  # ✅ 恢复训练：从指定 epoch 开始
        # ✅ 添加：记录 epoch 开始时间
        epoch_start_time = time.time()
        
        method.train()
        epoch_loss = 0.0
        num_batches = 0

        # 进度条
        if disable_tqdm:
            pbar = train_loader
        else:
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

                # 检查 loss 是否为 NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Warning: NaN/Inf loss detected at step {global_step}, skipping this batch")
                    continue  # 跳过这个 batch
                
                scaler.scale(loss).backward()
                
                # 检查梯度是否有 inf/nan（必须在 step 之前）
                scaler.unscale_(optimizer)
                
                # 检查梯度是否有 inf/nan
                grad_norm = torch.nn.utils.clip_grad_norm_(method.parameters(), max_norm=1.0)
                
                # 如果梯度有 inf/nan，scaler 会跳过 step
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = method.compute_loss(views)
                
                # 检查 loss 是否为 NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Warning: NaN/Inf loss detected at step {global_step}, skipping this batch")
                    continue  # 跳过这个 batch
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(method.parameters(), max_norm=1.0)
                
                optimizer.step()

            # 更新 EMA（如果有 teacher 网络）
            method.update_ema()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # 检查梯度（用于调试）
            if global_step % log_freq == 0 and not use_amp:
                # ✅ 性能优化：在 GPU 上计算梯度范数，最后只调用一次 .item()
                # 计算梯度范数（仅在非 AMP 模式下，避免影响性能）
                total_norm_sq = torch.tensor(0.0, device=device)
                for p in method.parameters():
                    if p.grad is not None:
                        param_norm_sq = p.grad.data.norm(2) ** 2
                        total_norm_sq = total_norm_sq + param_norm_sq
                total_norm = (total_norm_sq ** 0.5).item()  # 只在最后调用一次 .item()
                if total_norm > 0:
                    loss_dict["grad_norm"] = total_norm

            # Step 级别日志
            if global_step % log_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                if not disable_tqdm:
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
        
        # ✅ 添加：计算 epoch 耗时
        epoch_time = time.time() - epoch_start_time
        epoch_time_min = epoch_time / 60.0
        epoch_time_sec = epoch_time % 60

        print(f"\n📌 Epoch {epoch}/{epochs}:")
        print(f"   avg_loss = {avg_loss:.4f}")
        print(f"   lr = {current_lr:.3e}")
        print(f"   耗时 = {int(epoch_time_min)}分{int(epoch_time_sec)}秒 ({epoch_time:.2f}秒)")

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

        # 保存当前 epoch 的 checkpoint（仅在 save_freq 不为 None 时保存）
        if save_freq is not None and (epoch % save_freq == 0 or epoch == epochs):
            ckpt = {
                "epoch": epoch,
                "model_state_dict": method.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "global_step": global_step,
                "best_loss": best_loss,  # ✅ 添加：保存 best_loss 以便恢复训练
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
        
        # 保存 latest checkpoint（每个 epoch 都保存）
        latest_path = os.path.join(save_dir, "latest.pth")
        ckpt = {
            "epoch": epoch,
            "model_state_dict": method.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss,
            "global_step": global_step,
            "best_loss": best_loss,  # ✅ 添加：保存 best_loss 以便恢复训练
        }
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = (
                scheduler.scheduler.state_dict()
                if hasattr(scheduler, "scheduler")
                else scheduler.state_dict()
            )
        torch.save(ckpt, latest_path)
        if epoch == 1 or epoch % log_freq == 0:
            print(f"💾 更新 Latest 模型到 {latest_path}")

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
                "best_loss": best_loss,  # ✅ 添加：保存 best_loss
            }
            if scheduler is not None:
                ckpt["scheduler_state_dict"] = (
                    scheduler.scheduler.state_dict()
                    if hasattr(scheduler, "scheduler")
                    else scheduler.state_dict()
                )
            torch.save(ckpt, best_path)
            print(f"🏅 更新 Best 模型（loss={best_loss:.4f}）到 {best_path}")

            if use_wandb:
                wandb.run.summary["best_loss"] = best_loss
                wandb.run.summary["best_epoch"] = epoch
                wandb.save(best_path)
        else:
            epochs_without_improvement += 1
            if early_stop_patience is not None:
                print(f"⚠️  Loss 没有改善 ({epochs_without_improvement}/{early_stop_patience})")

        # 评估（每 eval_freq 个 epoch）
        if eval_enabled and epoch % eval_freq == 0:
            # ✅ 添加：记录评估开始时间
            eval_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch}: 开始评估...")
            print(f"{'='*60}")
            try:
                eval_results = evaluate_on_cub(
                    method=method,
                    cub_data_dir=eval_cub_data_dir,
                    device=device,
                    img_size=img_size,
                    batch_size=eval_batch_size,
                    num_workers=eval_num_workers,
                    eval_method=eval_method,
                    use_cls_token=eval_use_cls_token,
                    knn_k=eval_knn_k,
                    linear_probe_C=eval_linear_probe_C,
                    verbose=True,
                    disable_tqdm=disable_tqdm
                )
                
                # ✅ 添加：计算评估耗时
                eval_time = time.time() - eval_start_time
                eval_time_min = eval_time / 60.0
                eval_time_sec = eval_time % 60
                
                eval_accuracy = eval_results["accuracy"]
                print(f"\n✅ Epoch {epoch} 评估完成: {eval_method} accuracy = {eval_accuracy:.4f} ({eval_accuracy*100:.2f}%)")
                print(f"   评估耗时 = {int(eval_time_min)}分{int(eval_time_sec)}秒 ({eval_time:.2f}秒)")
                
                # 记录到 wandb
                if use_wandb:
                    wandb.log(
                        {
                            f"eval/{eval_method}_accuracy": eval_accuracy,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
            except Exception as e:
                print(f"⚠️  评估失败: {e}")
                import traceback
                traceback.print_exc()
        
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
    # 验证评估参数
    if args.eval_enabled:
        if args.eval_cub_data_dir is None:
            raise ValueError("--eval_cub_data_dir 必须提供（当 --eval_enabled 时）")
        from pathlib import Path
        cub_path = Path(args.eval_cub_data_dir)
        if not cub_path.exists():
            raise ValueError(f"CUB 数据路径不存在: {args.eval_cub_data_dir}")
        if not (cub_path / "train").exists() or not (cub_path / "val").exists():
            raise ValueError(f"CUB 数据路径格式不正确，应包含 train/ 和 val/ 文件夹: {args.eval_cub_data_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 设备: {device}")
    
    # 确定保存目录
    print(f"🔍 路径处理调试信息:")
    print(f"   传入的 --save_dir: {args.save_dir}")
    print(f"   传入的 --exp_name: {args.exp_name}")
    print(f"   当前工作目录: {os.getcwd()}")
    
    if args.save_dir is None:
        if args.exp_name:
            args.save_dir = os.path.join("./checkpoints", args.exp_name)
        else:
            # 如果没有提供exp_name，使用默认命名
            args.save_dir = os.path.join(
                "./checkpoints",
                f"{args.method}_{args.backbone_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    else:
        # 如果提供了 save_dir，且也提供了 exp_name，则组合路径
        if args.exp_name:
            args.save_dir = os.path.join(args.save_dir, args.exp_name)
            print(f"   组合后的路径（拼接后）: {args.save_dir}")
    
    # 转换为绝对路径，避免相对路径问题
    # 如果已经是绝对路径，os.path.abspath() 会保持不变
    original_save_dir = args.save_dir
    args.save_dir = os.path.abspath(args.save_dir)
    
    print(f"📁 Checkpoint 保存目录（最终）: {args.save_dir}")
    if original_save_dir != args.save_dir:
        print(f"   ⚠️  路径已从相对路径转换为绝对路径")
    print(f"   目录是否存在: {os.path.exists(args.save_dir)}")
    if not os.path.exists(args.save_dir):
        print(f"   ⚠️  目录不存在，将创建: {args.save_dir}")

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
    # 根据方法类型决定是否使用 multi-crop
    use_multi_crop = args.method.lower() in ["dino", "dinov2", "ibot"]
    num_local_crops = 8 if use_multi_crop else 0  # DINOv2 默认 8 个 local crops
    
    train_loader, _, _, two_view_aug = load_dino_data(
        dataset_type=args.dataset_type,     # "local" 或 "huggingface"
        dataset_root=args.dataset_root,     # 本地时使用
        dataset_name=args.dataset_name,     # HF 时使用
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_sample=args.train_sample,
        strength=args.aug_strength,
        method=args.method,  # 传递方法类型
        num_local_crops=num_local_crops,  # 传递 local crops 数量
    )

    # 构建方法配置
    method_config = {
        "proj_hidden_dim": args.proj_hidden_dim,
        "proj_output_dim": args.proj_output_dim,
        "temperature": args.temperature,
        "img_size": args.img_size,  # 传递给 backbone 构建函数，用于 ViT 的自定义图像尺寸
        "total_epochs": args.epochs,  # ✅ 修复：传递给 DINOv2 用于 momentum cosine 调度
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

    # ============================================================
    # 恢复训练：从 checkpoint 加载
    # ============================================================
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    
    if args.resume:
        print("\n" + "="*60)
        print(f"🔄 从 checkpoint 恢复训练: {args.resume}")
        print("="*60)
        
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint 文件不存在: {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location=device)
        
        # 加载模型
        method.load_state_dict(checkpoint["model_state_dict"])
        print("✅ 模型权重已加载")
        
        # 加载优化器
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ 优化器状态已加载")
        
        # 加载调度器
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            try:
                if hasattr(scheduler, "scheduler"):
                    scheduler.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                else:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("✅ 学习率调度器状态已加载")
            except Exception as e:
                print(f"⚠️  警告：无法加载调度器状态: {e}")
                print("   将使用新的调度器状态继续训练")
        
        # 加载训练状态
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1  # 从下一个 epoch 开始
            print(f"✅ 从 Epoch {start_epoch} 开始训练（已训练到 Epoch {checkpoint['epoch']}）")
        
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
            print(f"✅ Global step: {global_step}")
        
        if "avg_loss" in checkpoint:
            print(f"✅ 上一个 epoch 的平均 loss: {checkpoint['avg_loss']:.4f}")
        
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
            print(f"✅ Best loss: {best_loss:.4f}")
        
        # 如果 DINOv2 有 teacher 网络，需要确保 teacher 也被正确加载
        if hasattr(method, 'teacher_backbone') and hasattr(method, 'teacher_head'):
            print("✅ DINOv2 teacher 网络已随模型一起加载")
        
        print("="*60)
        print()

    # ============================================================
    # 训练前检查：验证训练代码是否正确
    # ============================================================
    print("\n" + "="*60)
    print("🔍 训练前检查：验证训练代码是否正确")
    print("="*60)
    
    # 检查优化器：backbone 参数是否在优化器中
    optimizer_param_ids = set(id(p) for group in optimizer.param_groups for p in group['params'])
    backbone_param_ids = set(id(p) for p in method.backbone.parameters())
    head_param_ids = set(id(p) for p in method.head.parameters())
    
    backbone_in_optimizer = len(backbone_param_ids & optimizer_param_ids) > 0
    head_in_optimizer = len(head_param_ids & optimizer_param_ids) > 0
    
    print(f"📊 优化器参数检查：")
    print(f"   Backbone 参数在优化器中: {'✅ 是' if backbone_in_optimizer else '❌ 否（这是严重问题！）'}")
    print(f"   Head 参数在优化器中: {'✅ 是' if head_in_optimizer else '❌ 否（这是严重问题！）'}")
    
    if not backbone_in_optimizer:
        print("\n⚠️  严重警告：Backbone 参数不在优化器中，不会被更新！")
        print("   这会导致训练无效，准确率不会提升！")
        print("   请检查代码，确保 backbone 参数被添加到优化器中。")
    
    # 检查梯度：backbone 是否有梯度
    print(f"\n📊 梯度检查：")
    method.train()
    dummy_batch = torch.randn(2, 3, args.img_size, args.img_size).to(device)
    views = torch.stack([dummy_batch, dummy_batch], dim=1)  # [2, 2, 3, H, W]
    
    optimizer.zero_grad()
    loss, _ = method.compute_loss(views)
    loss.backward()
    
    backbone_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in method.backbone.parameters())
    head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in method.head.parameters())
    
    print(f"   Backbone 有梯度: {'✅ 是' if backbone_has_grad else '❌ 否（这是严重问题！）'}")
    print(f"   Head 有梯度: {'✅ 是' if head_has_grad else '❌ 否（这是严重问题！）'}")
    
    if not backbone_has_grad:
        print("\n⚠️  严重警告：Backbone 没有梯度，梯度没有正确传播！")
        print("   这会导致训练无效，准确率不会提升！")
        print("   请检查代码，确保梯度能够传播到 backbone。")
    
    if backbone_in_optimizer and backbone_has_grad:
        print("\n✅ 训练代码检查通过：Backbone 会被正确更新")
    else:
        print("\n❌ 训练代码检查失败：存在问题，需要修复！")
    
    print("="*60)
    print()

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
        start_epoch=start_epoch,  # ✅ 添加：从指定 epoch 开始
        start_global_step=global_step,  # ✅ 添加：从指定 global_step 开始
        start_best_loss=best_loss,  # ✅ 添加：从指定 best_loss 开始
        # 评估参数
        eval_enabled=args.eval_enabled,
        eval_cub_data_dir=args.eval_cub_data_dir,
        eval_freq=args.eval_freq,
        eval_method=args.eval_method,
        eval_knn_k=args.eval_knn_k,
        eval_linear_probe_C=args.eval_linear_probe_C,
        eval_use_cls_token=args.eval_use_cls_token,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers,
        img_size=args.img_size,
        disable_tqdm=args.disable_tqdm,
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
        choices=["simclr", "moco", "byol", "dino", "dinov2", "ibot", "vicreg", "mae"],
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
        choices=["resnet50", "vit_s_16", "vit_b_16", "vit_s_14", "vit_b_14"],
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
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称（用于命名checkpoint目录，例如：dinov2_vitb16_96px）")
    parser.add_argument("--save_dir", type=str, default=None, help="保存目录（如果提供exp_name，会自动生成：./checkpoints/{exp_name}）")
    parser.add_argument("--save_freq", type=int, default=None, help="保存频率（每 N 个 epoch 保存一次 epoch 特定的 checkpoint，默认 None 表示只保存 latest 和 best）")
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
    
    # 恢复训练
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的 checkpoint 路径（例如：./checkpoints/exp_name/latest.pth 或 ./checkpoints/exp_name/epoch_010.pth）",
    )

    # 评估参数
    parser.add_argument(
        "--eval_enabled",
        action="store_true",
        help="是否启用评估（在 CUB-200-2011 上每 N 个 epoch 评估一次）",
    )
    parser.add_argument(
        "--eval_cub_data_dir",
        type=str,
        default=None,
        help="CUB-200-2011 数据文件夹路径（包含 train/val/test，eval_enabled=True 时必需）",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=2,
        help="评估频率（每 N 个 epoch 评估一次，默认 2）",
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="knn",
        choices=["knn", "linear_probe"],
        help="评估方法：knn 或 linear_probe",
    )
    parser.add_argument(
        "--eval_knn_k",
        type=int,
        default=20,
        help="k-NN 评估的 k 值",
    )
    parser.add_argument(
        "--eval_linear_probe_C",
        type=float,
        default=1.0,
        help="Linear Probe 的正则化强度",
    )
    parser.add_argument(
        "--eval_use_cls_token",
        action="store_true",
        help="是否使用 CLS token（仅 ViT）",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="评估时的批次大小",
    )
    parser.add_argument(
        "--eval_num_workers",
        type=int,
        default=4,
        help="评估时的数据加载线程数",
    )

    # 其他参数
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="禁用 tqdm 进度条（适用于非交互式环境或日志文件）",
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
