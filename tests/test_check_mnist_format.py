# 在项目根目录创建 test_check_mnist_format.py
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

# 加载MNIST数据（不做任何transform）
train_data = datasets.MNIST('./data/raw', train=True, download=False)

# 获取第一个样本
img, label = train_data[0]
img_array = np.array(img)

print(f'图像形状: {img_array.shape}')
print(f'像素值范围: [{img_array.min()}, {img_array.max()}]')
print(f'背景像素值(左上角10x10区域平均): {img_array[0:10, 0:10].mean():.1f}')
print(f'标签: {label}')

# 保存一些样本
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img, label = train_data[i]
    img_array = np.array(img)
    axes[i].imshow(img_array, cmap='gray')
    axes[i].set_title(f'Label: {label}\nBG: {img_array[0,0]:.0f}')
    axes[i].axis('off')

plt.savefig('mnist_samples.png')
print('\n已保存 mnist_samples.png')
print('\nMNIST格式说明：')
print('- 0 = 黑色（背景）')
print('- 255 = 白色（数字笔画）')

