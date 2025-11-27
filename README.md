# 手写数字识别项目

基于PyTorch的手写数字识别项目，使用MNIST数据集训练CNN和MLP模型。

## 项目结构

```
digit-recognition/
├── config/                         # 配置文件
│   └── config.yaml                 # 模型和训练参数配置
├── data/                           # 数据目录
│   ├── kaggle                      # kaggle测试数据
│   └── raw/                        # 原始数据（MNIST数据集）
├── models/                         # 训练好的模型
│   └── best_model.pth              # 最佳模型权重
├── notebooks/                      # Jupyter笔记本
│   ├── 01_data_exploration.ipynb   # 数据探索与可视化
│   ├── 02_model_experiments.ipynb  # 模型训练实验
│   └── 03_evaluation.ipynb         # 模型评估分析
├── src/                            # 源代码
│   ├── data/                       # 数据处理模块
│   │   ├── loader.py               # 数据加载
│   │   └── preprocessing.py        # 数据预处理
│   ├── models/                     # 模型定义
│   │   ├── cnn.py                  # 卷积神经网络
│   │   └── mlp.py                  # 多层感知机
│   └── training/                   # 训练模块
│       └── trainer.py              # 训练器
├── tests/                          # 测试代码
├── .gitignore                      # Git忽略文件
├── LICENSE                         # 许可证
├── README.md                       # 项目说明
├── requirements.txt                # 依赖包
├── prediction.py                   # 预测脚本
└── train.py                        # 训练脚本
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/AkiNeko11/digit-recognition.git
cd digit-recognition

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装依赖
# 默认直接执行安装的是cpu版本的torch，如需使用gpu版本的请先安装torch再执行
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认配置训练
python train.py

# 训练完成后会在models/目录下生成best_model.pth
```

### 3. 使用Jupyter Notebook

```bash
# 启动Jupyter Notebook
jupyter notebook

# 或使用JupyterLab
jupyter lab
```

然后打开`notebooks/`目录下的笔记本文件进行后续操作。

## 配置说明

在`config/config.yaml`中可以修改以下参数：

```yaml
data:
  batch_size: 128          # 批次大小
  val_split: 0.2           # 验证集分割
  test_path: "data/kaggle/digit-recognizer/test.csv"  # 测试数据路径
  
model:
  type: "cnn"              # 模型类型：cnn 或 mlp
  learning_rate: 0.001     # 学习率
  epochs: 10               # 训练轮数

output:
  save_path: "data/kaggle/digit-recognizer/submission.csv"  # 输出文件路径

prediction:
  batch_size: 128          # 预测批次大小
```

## 模型性能

- **CNN模型**：约695万参数，在MNIST测试集上准确率通常>99%
- **MLP模型**：约57万参数，在MNIST测试集上准确率通常>97%

## 依赖包

- torch>=2.0.0
- torchvision>=0.15.0
- torchaudio>=2.0.0
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- jupyter>=1.0.0
- tqdm>=4.62.0
- pyyaml>=6.0

## 许可证

[MIT License](LICENSE)