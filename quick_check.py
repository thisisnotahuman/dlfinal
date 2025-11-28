"""
快速检查：验证训练是否真的在更新 backbone
"""

import torch
import torch.nn as nn

def check_backbone_gradients(method, device):
    """检查 backbone 是否有梯度"""
    method.train()
    
    # 创建一个 dummy batch
    dummy_batch = torch.randn(2, 3, 96, 96).to(device)
    views = torch.stack([dummy_batch, dummy_batch], dim=1)  # [2, 2, 3, 96, 96]
    
    # 前向传播
    loss, _ = method.compute_loss(views)
    
    # 反向传播
    loss.backward()
    
    # 检查 backbone 参数是否有梯度
    backbone_has_grad = False
    backbone_param_count = 0
    for name, param in method.backbone.named_parameters():
        backbone_param_count += 1
        if param.grad is not None:
            backbone_has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"✅ Backbone 参数 '{name}' 有梯度，梯度范数: {grad_norm:.6f}")
            break
    
    if not backbone_has_grad:
        print(f"❌ 警告：检查了 {backbone_param_count} 个 backbone 参数，都没有梯度！")
        print("   这可能意味着 backbone 被冻结了，或者梯度没有正确传播")
    else:
        print(f"✅ Backbone 有梯度，训练应该正常")
    
    # 检查 head 参数是否有梯度
    head_has_grad = False
    head_param_count = 0
    for name, param in method.head.named_parameters():
        head_param_count += 1
        if param.grad is not None:
            head_has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"✅ Head 参数 '{name}' 有梯度，梯度范数: {grad_norm:.6f}")
            break
    
    if not head_has_grad:
        print(f"❌ 警告：检查了 {head_param_count} 个 head 参数，都没有梯度！")
    else:
        print(f"✅ Head 有梯度，训练应该正常")
    
    return backbone_has_grad and head_has_grad


def check_optimizer_params(optimizer, method):
    """检查优化器中是否包含 backbone 参数"""
    optimizer_param_ids = set(id(p) for group in optimizer.param_groups for p in group['params'])
    
    backbone_param_ids = set(id(p) for p in method.backbone.parameters())
    head_param_ids = set(id(p) for p in method.head.parameters())
    
    backbone_in_optimizer = len(backbone_param_ids & optimizer_param_ids) > 0
    head_in_optimizer = len(head_param_ids & optimizer_param_ids) > 0
    
    print(f"\n📊 优化器参数检查：")
    print(f"   Backbone 参数在优化器中: {'✅ 是' if backbone_in_optimizer else '❌ 否'}")
    print(f"   Head 参数在优化器中: {'✅ 是' if head_in_optimizer else '❌ 否'}")
    
    if not backbone_in_optimizer:
        print("   ⚠️  警告：Backbone 参数不在优化器中，不会被更新！")
    
    return backbone_in_optimizer and head_in_optimizer


if __name__ == "__main__":
    print("="*60)
    print("🔍 快速检查：训练代码是否正确")
    print("="*60)
    
    # 这里需要导入你的实际代码
    # 由于在服务器上运行，你需要修改这个脚本来适配你的环境
    print("\n⚠️  这个脚本需要在你的训练环境中运行")
    print("   请将以下代码添加到你的训练脚本中，在训练开始前检查：")
    print()
    print("""
    # 在构建 optimizer 之后，训练开始前添加：
    from quick_check import check_backbone_gradients, check_optimizer_params
    
    # 检查优化器
    check_optimizer_params(optimizer, method)
    
    # 检查梯度
    check_backbone_gradients(method, device)
    """)

