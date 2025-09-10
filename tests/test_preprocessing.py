# 在项目根目录创建 test_preprocessing.py
from src.data.loader import load_mnist_data
from src.data.preprocessing import visualize_samples, print_data_summary

if __name__ == "__main__":
    # 加载数据
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # 打印数据摘要
    print_data_summary(train_loader, test_loader)
    
    # 可视化样本
    visualize_samples(train_loader)