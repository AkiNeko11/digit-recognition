import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitMLP(nn.Module):
    """
    手写数字识别的多层感知机
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout_rate=0.5): 
        """
        Args:
            input_size=784: MNIST 图像为 28*28 像素，展平后为 784 维向量。
            hidden_sizes=[512, 256, 128]：定义了三个隐藏层的神经元数量，构成多层全连接网络。
            num_classes=10: MNIST 有 10 个类别(数字 0-9)。
            dropout_rate=0.5: Dropout 正则化概率，随机丢弃 50% 的神经元以防止过拟合。
        """

        super(DigitMLP, self).__init__()
        
        # 构建全连接层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))     # 添加全连接层，将前一层的输出映射到当前层的神经元数量。
            layers.append(nn.ReLU())                             # 添加 ReLU 激活函数，增加模型的非线性能力。
            layers.append(nn.Dropout(dropout_rate))              # 添加 Dropout 层，随机丢弃 50% 的神经元以防止过拟合。
            prev_size = hidden_size
        
        
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))         # 添加全连接层，将前一层的输出映射到输出类别数。
        
        self.network = nn.Sequential(*layers)                    # 将所有层组合成序列模型。
        
    def forward(self, x):
        # 展平输入
        """
        展平输入：
        x.view(x.size(0), -1) 将输入张量展平。
            x.size(0) 是批量大小(batch size)。-1 表示自动计算剩余维度。
            对于 MNIST,输入 x 的形状通常是 (batch_size, 1, 28, 28)(张量格式，通道数为 1)，展平后变为 (batch_size, 784)。

        通过网络：
            展平后的张量通过 self.network(即 nn.Sequential 定义的层)进行前向传播，输出形状为 (batch_size, num_classes)。
        """
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_model_info(self):
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")    #  512*256+512+  256*128+128  +128*64+64  +64*32+32  +32*10+10= 567,434
        print(f"可训练参数: {trainable_params:,}")    #  512*256+512+  256*128+128  +128*64+64  +64*32+32  +32*10+10= 567,434
        
        return total_params, trainable_params