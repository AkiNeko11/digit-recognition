this is a repo for handwritten digits recognition

using data from mnist

digit-recognition/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据（MNIST数据集）
│   ├── processed/                 # 预处理后的数据
│   └── .gitkeep                  # 保持空目录在git中
├── notebooks/                     # Jupyter笔记本
│   ├── 01_data_exploration.ipynb # 数据探索与可视化
│   ├── 02_model_experiments.ipynb # 模型实验
│   └── 03_evaluation.ipynb       # 模型评估
├── src/                          # 源代码
│   ├── __init__.py
│   ├── data/                     # 数据处理模块
│   │   ├── __init__.py
│   │   ├── loader.py             # 数据加载
│   │   └── preprocessing.py      # 数据预处理
│   ├── models/                   # 模型定义
│   │   ├── __init__.py
│   │   ├── cnn.py               # CNN模型
│   │   └── mlp.py               # 多层感知机模型
│   ├── training/                 # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py           # 训练器
│   │   └── callbacks.py         # 训练回调
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── visualization.py     # 可视化工具
│       └── metrics.py           # 评估指标
├── tests/                        # 测试代码
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   └── test_training/
├── models/                       # 保存训练好的模型
│   └── .gitkeep
├── results/                      # 实验结果
│   ├── plots/                    # 图表
│   └── logs/                     # 训练日志
├── config/                       # 配置文件
│   └── config.yaml
├── requirements.txt              # 依赖包
├── setup.py                     # 安装脚本
├── .gitignore                   # Git忽略文件
├── README.md                    # 项目说明
└── LICENSE                      # 许可证