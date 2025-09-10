import torch
import yaml
from src.models.cnn import DigitCNN
from src.models.mlp import DigitMLP
from src.training.trainer import Trainer, create_data_loaders

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 加载配置
    config = load_config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=config['data']['batch_size'],
        val_split=config['data']['val_split']  
    )
    
    # 选择模型
    if config['model']['type'] == 'cnn':
        model = DigitCNN()
    else:
        model = DigitMLP()
    print(f"模型: {model.__class__.__name__}")
    
    # 创建训练器
    trainer = Trainer(model, device, learning_rate=config['model']['learning_rate'])
    
    # 开始训练
    best_acc = trainer.train(train_loader, val_loader, epochs=config['model']['epochs'])
    
    # 测试集评估
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")

if __name__ == "__main__":
    main()