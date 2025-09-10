import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    """
    手写数字识别的卷积神经网络
    """
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        
        # 卷积层1 输入通道数为1，输出通道数为32，卷积核大小为3*3，填充为1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 卷积层2 输入通道数为32，输出通道数为64，卷积核大小为3*3，填充为1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层 池化核大小为2，步长为2，每次减半图像尺寸，例如28*28 -> 14*14 -> 7*7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层3 输入通道数为64，输出通道数为128，卷积核大小为3*3，填充为1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 卷积层4 输入通道数为128，输出通道数为256，卷积核大小为3*3，填充为1
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 全连接层 输入维度为256 * 7 * 7，输出维度为512，dropout率为0.5
        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # 由于卷积层1和卷积层2的池化层，图像尺寸从 28*28 -> 14*14 -> 7*7
        self.dropout1 = nn.Dropout(0.5)         # 添加 Dropout 层，随机丢弃 50% 的神经元以防止过拟合。
        self.fc2 = nn.Linear(512, 256)          # 添加全连接层，将前一层的输出映射到当前层的神经元数量。
        self.dropout2 = nn.Dropout(0.5)         # 添加 Dropout 层，随机丢弃 50% 的神经元以防止过拟合。  
        self.fc3 = nn.Linear(256, num_classes)  # 添加全连接层，将前一层的输出映射到输出类别数。
        
    def forward(self, x):
        """
        前向传播
        """
        # 卷积层1 + 批归一化 + ReLU + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积层2 + 批归一化 + ReLU + 池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积层3 + 批归一化 + ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 卷积层4 + 批归一化 + ReLU
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self):
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")    #  6,944,738
        print(f"可训练参数: {trainable_params:,}")    #  6,944,738

        """
        层参数量
        conv1:320           (32*3*3*1+32)=320 
        bn1:64              (32*2)=64
        conv2:18,496        (32*64*3*3+64)=18,496
        bn2:128             (64*2)=128
        conv3:73,856        (64*128*3*3+128)=73,856
        bn3:256             (128*2)=256
        conv4:295,168       (128*256*3*3+256)=295,168
        bn4:512             (256*2)=512
        fc1:6,423,040       (256*7*7*512+512)=6,423,040
        fc2:131,328         (512*256+256)=131,328
        fc3:2,570           (256*10+10)=2,570 
        总计:6,944,738
        """
        
        return total_params, trainable_params