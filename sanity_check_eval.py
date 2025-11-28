"""
Sanity Check: 验证评估 pipeline 是否正确
1. 用 ImageNet 预训练的模型测试评估 pipeline
2. 对比随机初始化的模型
"""

import torch
import torch.nn as nn
from eval import evaluate_on_cub
from utils import build_backbone

def test_with_pretrained_model():
    """使用预训练的 ViT-S/16 测试评估 pipeline（如果有的话）"""
    print("="*60)
    print("🔍 Sanity Check 1: 预训练模型（如果有）")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 构建预训练的 ViT-S/16
        # 注意：ViT-S/16 可能没有标准的 ImageNet 预训练权重
        # 代码会尝试加载 DINOv2 权重，如果失败则使用随机初始化
        backbone, feat_dim = build_backbone(
            backbone_type="vit_s_16",
            image_size=96,
            pretrained=True  # 尝试使用预训练权重
        )
        backbone = backbone.to(device)
        
        # 创建一个简单的包装类，模拟 BaseSSLMethod
        class DummyMethod:
            def __init__(self, backbone):
                self.backbone = backbone
            
            def get_encoder(self):
                return self.backbone
        
        method = DummyMethod(backbone)
        
        # 评估
        results = evaluate_on_cub(
            method=method,
            cub_data_dir="/root/dl/eval_data/data",
            device=device,
            img_size=96,
            batch_size=256,
            num_workers=4,
            eval_method="linear_probe",
            disable_tqdm=False
        )
        
        print(f"\n✅ 预训练模型结果: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        # 预期：如果有真正的预训练权重，应该能达到 30%+
        # 如果没有预训练权重（使用随机初始化），结果会接近随机猜测
        if results['accuracy'] > 0.3:
            print("✅ 评估 pipeline 正常！预训练模型表现良好。")
            return results
        elif results['accuracy'] > 0.01:
            print("⚠️  预训练模型表现中等，可能是权重加载失败，使用了随机初始化。")
            return results
        else:
            print("⚠️  预训练模型表现异常，可能是评估 pipeline 有问题！")
            return results
            
    except Exception as e:
        print(f"⚠️  预训练模型测试跳过: {e}")
        print("   这可能是正常的（ViT-S/16 可能没有标准的预训练权重）")
        return None


def test_with_random_model():
    """使用随机初始化的模型测试"""
    print("\n" + "="*60)
    print("🔍 Sanity Check 2: 随机初始化模型")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建随机初始化的 ViT-S/16
    backbone, feat_dim = build_backbone(
        backbone_type="vit_s_16",
        image_size=96,
        pretrained=False  # 不使用预训练权重
    )
    backbone = backbone.to(device)
    
    # 创建一个简单的包装类
    class DummyMethod:
        def __init__(self, backbone):
            self.backbone = backbone
        
        def get_encoder(self):
            return self.backbone
    
    method = DummyMethod(backbone)
    
    # 评估
    results = evaluate_on_cub(
        method=method,
        cub_data_dir="/root/dl/eval_data/data",
        device=device,
        img_size=96,
        batch_size=256,
        num_workers=4,
        eval_method="linear_probe",
        disable_tqdm=False
    )
    
    print(f"\n✅ 随机初始化模型结果: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # 预期：随机初始化的模型应该接近随机猜测（0.5%）
    expected_random = 1.0 / 200  # CUB-200 有 200 类
    if abs(results['accuracy'] - expected_random) < 0.01:
        print(f"✅ 随机模型表现符合预期（接近 {expected_random*100:.2f}%）")
    else:
        print(f"⚠️  随机模型表现: {results['accuracy']*100:.2f}%，预期约 {expected_random*100:.2f}%")
    
    return results


def compare_with_trained_model(checkpoint_path):
    """对比训练后的模型"""
    print("\n" + "="*60)
    print("🔍 Sanity Check 3: 训练后的模型")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载训练后的模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 这里需要根据你的 checkpoint 格式来加载
    # 假设 checkpoint 包含 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 构建模型（需要根据你的实际代码调整）
    from model import build_method
    from utils import build_optimizer
    
    # 这里需要根据你的实际配置来构建
    # 暂时跳过，因为需要完整的配置
    
    print("⚠️  需要根据实际的 checkpoint 格式来加载模型")
    print(f"   Checkpoint 路径: {checkpoint_path}")
    print(f"   Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")


if __name__ == "__main__":
    print("🚀 开始 Sanity Check...")
    print()
    
    # Check 1: ImageNet 预训练模型
    try:
        pretrained_results = test_with_pretrained_model()
    except Exception as e:
        print(f"❌ ImageNet 预训练模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # Check 2: 随机初始化模型
    try:
        random_results = test_with_random_model()
    except Exception as e:
        print(f"❌ 随机初始化模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print("\n" + "="*60)
    print("📊 总结")
    print("="*60)
    print("1. 如果 ImageNet 预训练模型表现良好（>30%），说明评估 pipeline 正常")
    print("2. 如果随机模型接近随机猜测（~0.5%），说明评估 pipeline 正常")
    print("3. 如果两者都正常，但你的训练模型只有 1-2%，说明训练确实还没学到有用特征")
    print("="*60)

