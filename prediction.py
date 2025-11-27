import torch
import os
import pandas as pd
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn import DigitCNN
from src.models.mlp import DigitMLP

class DigitPredictor:
    def __init__(self, config_path='config/config.yaml'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model()
        self.load_model()
    
    def load_config(self, config_path):
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置加载失败: {e}")
            return {}
    
    def create_model(self):
        """创建模型"""
        model_type = self.config.get('model', {}).get('type', 'cnn')
        if model_type == 'cnn':
            model = DigitCNN()
        elif model_type == 'mlp':
            model = DigitMLP()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        print(f"创建 {model_type.upper()} 模型")
        return model
    
    def load_model(self):
        """加载模型权重"""
        model_path = self.config.get('model', {}).get('save_path', 'models/best_model.pth')
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def preprocess_data(self, csv_path):
        """预处理数据"""
        # 读取数据
        data = pd.read_csv(csv_path,header=0)
        print(f"读取数据: {data.shape}")
        
        # 转换为numpy并归一化
        pixels = data.values.astype(np.float32) / 255.0
        
        # 根据模型类型reshape
        model_type = self.config.get('model', {}).get('type', 'cnn')
        if model_type == 'cnn':
            pixels = pixels.reshape(-1, 1, 28, 28)
        else:
            pixels = pixels.reshape(-1, 784)
        
        return pixels, data.index + 1
    
    def predict(self, test_csv_path, output_csv_path=None):
        """执行预测"""
        if output_csv_path is None:
            output_csv_path = self.config.get('output', {}).get('save_path', 'submission.csv')
        
        # 预处理
        test_data, image_ids = self.preprocess_data(test_csv_path)
        
        # 创建DataLoader
        batch_size = self.config.get('prediction', {}).get('batch_size', 128)
        dataset = TensorDataset(torch.from_numpy(test_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 预测
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # 保存结果
        results_df = pd.DataFrame({
            'ImageId': image_ids,
            'Label': predictions
        })
        
        results_df.to_csv(output_csv_path, index=False)
        print(f"预测完成! 结果保存至: {output_csv_path}")
        print(f"预测样本数: {len(predictions)}")
        
        return results_df

def main():
    predictor = DigitPredictor()
    
    test_path = predictor.config.get('data', {}).get('test_path', 'data/test.csv')
    if not os.path.exists(test_path):
        print(f"测试数据不存在: {test_path}")
        return
    
    results = predictor.predict(test_path)

if __name__ == "__main__":
    main()