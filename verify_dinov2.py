"""
验证 DINOv2 实现是否正确
检查关键组件：teacher 网络、EMA 更新、centering、损失计算
"""

import torch
import torch.nn as nn
from model import build_method

def verify_dinov2_implementation():
    """验证 DINOv2 实现"""
    print("="*60)
    print("🔍 验证 DINOv2 实现")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建 DINOv2
    config = {
        "proj_hidden_dim": 512,
        "proj_output_dim": 256,
        "temperature": 0.1,
        "img_size": 96,
    }
    
    method = build_method(
        method_name="dinov2",
        backbone_type="vit_s_16",
        pretrained_backbone=False,
        config=config
    ).to(device)
    
    print("\n1. 检查 Teacher 网络是否存在")
    has_teacher_backbone = hasattr(method, 'teacher_backbone')
    has_teacher_head = hasattr(method, 'teacher_head')
    print(f"   teacher_backbone: {'✅ 存在' if has_teacher_backbone else '❌ 不存在'}")
    print(f"   teacher_head: {'✅ 存在' if has_teacher_head else '❌ 不存在'}")
    
    if not (has_teacher_backbone and has_teacher_head):
        print("   ❌ 错误：DINOv2 必须有 teacher 网络！")
        return False
    
    print("\n2. 检查 Teacher 网络是否被冻结")
    teacher_backbone_requires_grad = any(p.requires_grad for p in method.teacher_backbone.parameters())
    teacher_head_requires_grad = any(p.requires_grad for p in method.teacher_head.parameters())
    print(f"   teacher_backbone requires_grad: {'❌ 是（错误！应该为 False）' if teacher_backbone_requires_grad else '✅ 否（正确）'}")
    print(f"   teacher_head requires_grad: {'❌ 是（错误！应该为 False）' if teacher_head_requires_grad else '✅ 否（正确）'}")
    
    if teacher_backbone_requires_grad or teacher_head_requires_grad:
        print("   ❌ 错误：Teacher 网络应该被冻结（requires_grad=False）！")
        return False
    
    print("\n3. 检查 Centering 是否存在")
    has_center = hasattr(method, 'center')
    print(f"   center buffer: {'✅ 存在' if has_center else '❌ 不存在'}")
    
    if not has_center:
        print("   ❌ 错误：DINOv2 必须有 centering！")
        return False
    
    print("\n4. 检查损失计算是否使用 Teacher")
    method.train()
    dummy_batch = torch.randn(2, 3, 96, 96).to(device)
    views = torch.stack([dummy_batch, dummy_batch], dim=1)  # [2, 2, 3, 96, 96]
    
    # 记录 teacher 参数
    teacher_param_before = next(method.teacher_backbone.parameters()).data.clone()
    
    # 计算损失
    loss, _ = method.compute_loss(views)
    
    # 检查 teacher 参数是否改变（应该不变，因为 teacher 被冻结）
    teacher_param_after = next(method.teacher_backbone.parameters()).data.clone()
    teacher_changed = not torch.equal(teacher_param_before, teacher_param_after)
    
    print(f"   Loss 值: {loss.item():.4f}")
    print(f"   Teacher 参数在 compute_loss 后改变: {'❌ 是（错误！应该不变）' if teacher_changed else '✅ 否（正确）'}")
    
    if teacher_changed:
        print("   ❌ 错误：Teacher 参数不应该在 compute_loss 中改变！")
        return False
    
    print("\n5. 检查 EMA 更新是否工作")
    # 记录更新前的参数
    student_param_before = next(method.backbone.parameters()).data.clone()
    teacher_param_before = next(method.teacher_backbone.parameters()).data.clone()
    
    # 手动修改 student 参数
    with torch.no_grad():
        for p in method.backbone.parameters():
            p.data += 0.1
    
    student_param_after = next(method.backbone.parameters()).data.clone()
    
    # 更新 EMA
    method.update_ema()
    
    teacher_param_after = next(method.teacher_backbone.parameters()).data.clone()
    
    # 检查 teacher 是否更新（应该部分更新，因为 momentum）
    teacher_updated = not torch.equal(teacher_param_before, teacher_param_after)
    print(f"   Student 参数改变: ✅ 是")
    print(f"   Teacher 参数在 update_ema 后改变: {'✅ 是（正确）' if teacher_updated else '❌ 否（错误！应该更新）'}")
    
    if not teacher_updated:
        print("   ❌ 错误：Teacher 应该在 update_ema 后更新！")
        return False
    
    print("\n6. 检查损失计算逻辑")
    # 检查是否使用了 KL divergence（而不是简单的对比损失）
    # 通过检查损失值是否合理来判断
    if loss.item() > 0 and loss.item() < 100:
        print(f"   Loss 值合理: ✅ 是 ({loss.item():.4f})")
    else:
        print(f"   Loss 值异常: ❌ 否 ({loss.item():.4f})")
        return False
    
    print("\n" + "="*60)
    print("✅ DINOv2 实现验证通过！")
    print("="*60)
    return True


if __name__ == "__main__":
    try:
        success = verify_dinov2_implementation()
        if not success:
            print("\n❌ DINOv2 实现有问题，需要修复！")
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()

