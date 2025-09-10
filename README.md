this is a repo for handwritten digits recognition

using data from mnist

# 项目结构

```bash

digit-recognition/
├── config/                    # 配置文件
│   └── config.yaml
├── data/                      # 数据目录
│   └── raw/                   # 原始数据（MNIST数据集）
│       └── MNIST/raw/         # MNIST数据文件
├── models/                    # 训练好的模型
│   └── best_model.pth
├── notebooks/                 # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_evaluation.ipynb
├── src/                       # 源代码
│   ├── __init__.py
│   ├── data/                  # 数据处理模块
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── models/                # 模型定义
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   └── mlp.py
│   └── training/              # 训练模块
│       ├── __init__.py
│       └── trainer.py
├── tests/                     # 测试代码
│   ├── __init__.py
│   ├── test_cuda_device.py
│   ├── test_data_loading.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── venv/                      # 虚拟环境（不提交到git）
├── .gitignore                 # Git忽略文件
├── LICENSE                    # 许可证
├── README.md                  # 项目说明
├── requirements.txt           # 依赖包
└── train.py                   # 训练脚本

```