# 在项目根目录创建 test_data_loading.py
from src.data.loader import load_mnist_data, get_data_info

if __name__ == "__main__":
    # 加载数据
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # 打印数据信息
    print("训练集信息:")
    get_data_info(train_loader)
    
    print("\n测试集信息:")
    get_data_info(test_loader)
    
    # 查看一个批次的数据
    for images, labels in train_loader:
        print(f"\n图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"图像数据类型: {images.dtype}")
        print(f"标签数据类型: {labels.dtype}")
        print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        break