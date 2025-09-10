import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=128, data_dir='./data/raw'):
    """
    加载MNIST数据集
    
    Args:
        batch_size: 批次大小
        data_dir: 数据存储目录
    
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    # 定义数据变换（暂时只做归一化）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def get_data_info(data_loader):
    """
    获取数据集基本信息
    """
    total_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    num_batches = len(data_loader)
    
    print(f"数据集大小: {total_samples}")
    print(f"批次大小: {batch_size}")
    print(f"批次数: {num_batches}")
    
    return total_samples, batch_size, num_batches