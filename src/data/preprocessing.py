import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


# 设置中文字体，解决中文乱码问题
plt.rcParams['font.sans-serif']	=	['SimHei',	'Microsoft	YaHei']
plt.rcParams['axes.unicode_minus']	=	False
 

def get_data_transforms():
    """
    获取数据变换配置
    
    Returns:
        train_transform: 训练时的数据变换
        test_transform: 测试时的数据变换
    """
    # 训练时的数据变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST的均值和标准差
        # 可以添加数据增强
        # transforms.RandomRotation(10),      # 旋转10度   
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 仿射变换 degrees=0：表示不进行旋转；translate=(0.1, 0.1)：表示在水平和垂直方向上最多平移图像宽/高的 10%

    ])
    
    # 测试时的数据变换（不包含数据增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    return train_transform, test_transform

def visualize_samples(data_loader, num_samples=8):
    """
    可视化数据样本
    
    Args:
        data_loader: 数据加载器
        num_samples: 要显示的样本数量
    """
    # 获取一个批次的数据
    images, labels = next(iter(data_loader))
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # 反归一化显示
        img = images[i].squeeze() * 0.3081 + 0.1307
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'标签: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_class_distribution(data_loader):
    """
    获取类别分布信息
    
    Args:
        data_loader: 数据加载器
    
    Returns:
        class_counts: 每个类别的样本数量
    """
    class_counts = torch.zeros(10)
    
    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1
    
    return class_counts

def print_data_summary(train_loader, test_loader):
    """
    打印数据摘要信息
    """
    print("=== Data Summary ===")  # 改为英文
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Total samples: {len(train_loader.dataset) + len(test_loader.dataset)}")
    
    
    # 获取类别分布
    train_class_counts = get_class_distribution(train_loader)
    test_class_counts = get_class_distribution(test_loader)
    
    print("\n=== Class Distribution ===")  # 改为英文
    print("Class\tTrain\tTest\tTotal")
    print("-" * 30)
    for i in range(10):
        train_count = int(train_class_counts[i].item())
        test_count = int(test_class_counts[i].item())
        total = train_count + test_count
        print(f"{i}\t{train_count}\t{test_count}\t{total}")