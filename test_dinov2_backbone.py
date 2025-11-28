"""
测试 DINOv2 backbone 是否能正确处理不同尺寸的输入
"""
import torch
import torch.nn as nn

def test_dinov2_backbone():
    """测试 DINOv2 官方 backbone 是否支持动态尺寸"""
    print("=" * 60)
    print("测试 DINOv2 官方 ViT 是否支持动态尺寸")
    print("=" * 60)
    
    try:
        # 加载 DINOv2 官方模型
        print("\n1. 加载 DINOv2 ViT-S/14...")
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # 提取 backbone
        if hasattr(dinov2_model, 'student') and hasattr(dinov2_model.student, 'backbone'):
            backbone = dinov2_model.student.backbone
        elif hasattr(dinov2_model, 'backbone'):
            backbone = dinov2_model.backbone
        else:
            backbone = dinov2_model
        
        print(f"   ✅ Backbone 类型: {type(backbone)}")
        
        # 检查是否有 interpolate_pos_encoding 方法
        print("\n2. 检查是否支持动态尺寸...")
        has_interpolate = hasattr(backbone, 'interpolate_pos_encoding') or \
                         (hasattr(backbone, 'forward') and 'interpolate' in str(backbone.forward))
        
        # 测试不同尺寸的输入
        print("\n3. 测试不同尺寸的输入...")
        test_sizes = [(224, 224), (96, 96), (112, 112)]
        
        for h, w in test_sizes:
            try:
                x = torch.randn(1, 3, h, w)
                with torch.no_grad():
                    out = backbone(x)
                print(f"   ✅ {h}×{w}: 输出形状 {out.shape}")
            except Exception as e:
                print(f"   ❌ {h}×{w}: 失败 - {e}")
        
        # 检查 patch_embed 的 patch_size
        print("\n4. 检查 patch_embed 配置...")
        if hasattr(backbone, 'patch_embed'):
            patch_embed = backbone.patch_embed
            print(f"   PatchEmbed 类型: {type(patch_embed)}")
            if hasattr(patch_embed, 'patch_size'):
                print(f"   Patch size: {patch_embed.patch_size}")
            if hasattr(patch_embed, 'img_size'):
                print(f"   Image size (如果有): {getattr(patch_embed, 'img_size', 'N/A')}")
        
        # 检查 pos_embed
        print("\n5. 检查位置编码...")
        if hasattr(backbone, 'pos_embed'):
            pos_embed = backbone.pos_embed
            print(f"   Pos embed 形状: {pos_embed.shape}")
            print(f"   预期 patch 数量: {pos_embed.shape[1] - 1} (不包括 CLS token)")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dinov2_backbone()

