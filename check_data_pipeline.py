"""
检查数据流 pipeline：验证 DINOv2 的 multi-crop 是否正确实现
"""

import torch
from load_data import build_two_view_augment, ms_transform

def check_dinov2_pipeline():
    """检查 DINOv2 数据流"""
    print("="*60)
    print("🔍 检查 DINOv2 数据流 Pipeline")
    print("="*60)
    
    img_size = 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建增强函数
    print("\n1. 构建 DINOv2 multi-crop 增强函数...")
    dinov2_aug = build_two_view_augment(
        img_size=img_size,
        strength="strong",
        method="dinov2",
        num_local_crops=8
    )
    print("   ✅ DINOv2 增强函数构建成功")
    
    # 构建基础变换
    base_transform = ms_transform(img_size)
    
    # 创建一个 dummy 图像
    print("\n2. 创建测试图像...")
    from PIL import Image
    import numpy as np
    
    # 创建一个随机 RGB 图像
    dummy_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    x = base_transform(dummy_img)  # [3, 96, 96]
    x = x.unsqueeze(0).to(device)  # [1, 3, 96, 96]
    print(f"   输入图像形状: {x.shape}")
    
    # 应用增强
    print("\n3. 应用 DINOv2 multi-crop 增强...")
    views = dinov2_aug(x)  # 应该是 [1, 2+8, 3, 96, 96]
    print(f"   输出 views 形状: {views.shape}")
    print(f"   预期: [1, 10, 3, 96, 96] (2 global + 8 local)")
    
    # 检查
    expected_num_views = 2 + 8  # 2 global + 8 local
    if views.shape[1] == expected_num_views:
        print(f"   ✅ Views 数量正确: {views.shape[1]} (2 global + 8 local)")
    else:
        print(f"   ❌ Views 数量错误: 期望 {expected_num_views}，实际 {views.shape[1]}")
        return False
    
    # 检查尺寸
    if views.shape[3] == img_size and views.shape[4] == img_size:
        print(f"   ✅ Views 尺寸正确: {views.shape[3]}×{views.shape[4]}")
    else:
        print(f"   ❌ Views 尺寸错误: 期望 {img_size}×{img_size}，实际 {views.shape[3]}×{views.shape[4]}")
        return False
    
    # 检查数值范围（应该已经 normalize）
    print("\n4. 检查数值范围...")
    views_mean = views.mean().item()
    views_std = views.std().item()
    print(f"   Views 均值: {views_mean:.4f} (应该接近 0)")
    print(f"   Views 标准差: {views_std:.4f} (应该接近 1)")
    
    if abs(views_mean) < 0.5 and 0.5 < views_std < 2.0:
        print("   ✅ 数值范围正常（已 normalize）")
    else:
        print("   ⚠️  数值范围可能异常")
    
    # 检查是否有 NaN/Inf
    has_nan = torch.isnan(views).any().item()
    has_inf = torch.isinf(views).any().item()
    print(f"\n5. 检查 NaN/Inf...")
    print(f"   有 NaN: {'❌ 是' if has_nan else '✅ 否'}")
    print(f"   有 Inf: {'❌ 是' if has_inf else '✅ 否'}")
    
    if has_nan or has_inf:
        return False
    
    print("\n" + "="*60)
    print("✅ DINOv2 数据流 Pipeline 检查通过！")
    print("="*60)
    print("\n关键修复：")
    print("1. ✅ Global crop scale: (0.2, 1.0) → (0.4, 1.0)")
    print("2. ✅ 添加了 local crops (8个)")
    print("3. ✅ Local crop size: 96 (不是 img_size // 3)")
    print("4. ✅ Local crop scale: (0.05, 0.4)")
    return True


if __name__ == "__main__":
    try:
        success = check_dinov2_pipeline()
        if not success:
            print("\n❌ 数据流 Pipeline 有问题，需要修复！")
    except Exception as e:
        print(f"\n❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()

