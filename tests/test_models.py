import torch
from src.models.cnn import DigitCNN
from src.models.mlp import DigitMLP

if __name__ == "__main__":
    print("测试CNN模型:")
    cnn_model = DigitCNN()
    cnn_model.get_model_info()
    
    # 测试前向传播
    x = torch.randn(1, 1, 28, 28)
    output = cnn_model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n" + "="*50)
    
    print("测试MLP模型:")
    mlp_model = DigitMLP()
    mlp_model.get_model_info()
    
    # 测试前向传播
    x = torch.randn(1, 1, 28, 28)
    output = mlp_model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")