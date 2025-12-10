import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
from PIL import Image, ImageDraw
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import DigitCNN
from src.models.mlp import DigitMLP
import yaml


class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别 - MNIST 28x28")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # 加载模型
        self.load_model()
        
        # 画布设置 - 使用280x280显示，对应28x28的MNIST格式
        self.display_size = 280  # 显示大小（28x10）
        self.mnist_size = 28     # MNIST图像大小
        self.scale_factor = self.display_size // self.mnist_size  # 缩放因子 = 10
        self.brush_size = self.scale_factor  # 笔刷大小对应1个MNIST像素
        
        # 创建28x28的PIL图像用于实际绘制（MNIST格式）
        self.image = Image.new("L", (self.mnist_size, self.mnist_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # 是否显示网格
        self.show_grid = False
        
        # 设置UI（必须在show_grid初始化之后）
        self.setup_ui()
        
    def load_model(self):
        """加载训练好的模型"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'config', 'config.yaml')
        
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"配置加载失败: {e}")
            self.config = {'model': {'type': 'mlp', 'save_path': 'models/best_model.pth'}}
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model_type = self.config.get('model', {}).get('type', 'mlp')
        if model_type == 'cnn':
            self.model = DigitCNN()
        else:
            self.model = DigitMLP()
        
        # 加载权重
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   self.config.get('model', {}).get('save_path', 'models/best_model.pth'))
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"模型加载成功: {model_type.upper()}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        self.model_type = model_type
    
    def draw_grid(self):
        """绘制28x28网格线（辅助显示）"""
        if not self.show_grid:
            return
        for i in range(1, self.mnist_size):
            pos = i * self.scale_factor
            # 垂直线
            self.canvas.create_line(pos, 0, pos, self.display_size, 
                                   fill='#333333', tags='grid')
            # 水平线
            self.canvas.create_line(0, pos, self.display_size, pos, 
                                   fill='#333333', tags='grid')
    
    def toggle_grid(self):
        """切换网格显示"""
        self.show_grid = not self.show_grid
        if self.show_grid:
            self.draw_grid()
        else:
            self.canvas.delete('grid')
        
    def setup_ui(self):
        """设置用户界面"""
        # 主容器
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧 - 画布区域
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, padx=10)
        
        # 标题
        title_label = tk.Label(left_frame, text="在下方画板上绘制数字", 
                               font=("Arial", 14, "bold"), bg='#f0f0f0')
        title_label.pack(pady=5)
        
        # 说明文字
        info_label = tk.Label(left_frame, 
                             text="实际在 28×28 像素上绘制\n显示放大10倍便于操作", 
                             font=("Arial", 9), bg='#f0f0f0', fg='#666')
        info_label.pack(pady=2)
        
        # 画布 - 显示280x280，但实际对应28x28
        self.canvas = tk.Canvas(left_frame, width=self.display_size, height=self.display_size,
                               bg='black', cursor='circle', highlightthickness=1, highlightbackground='gray')
        self.canvas.pack(pady=5)
        
        # 绘制网格线（可选，显示28x28的像素边界）
        self.draw_grid()
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # 按钮框架
        button_frame = tk.Frame(left_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # 清除按钮
        clear_btn = tk.Button(button_frame, text="清除", command=self.clear_canvas,
                             font=("Arial", 12), bg='#ff6b6b', fg='white',
                             padx=20, pady=10, cursor='hand2')
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 识别按钮
        predict_btn = tk.Button(button_frame, text="识别", command=self.predict,
                               font=("Arial", 12), bg='#4ecdc4', fg='white',
                               padx=20, pady=10, cursor='hand2')
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # 保存图像按钮（用于调试）
        save_btn = tk.Button(button_frame, text="保存", command=self.save_image,
                            font=("Arial", 12), bg='#95a5a6', fg='white',
                            padx=20, pady=10, cursor='hand2')
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # 网格切换按钮
        grid_btn = tk.Button(button_frame, text="网格", command=self.toggle_grid,
                            font=("Arial", 12), bg='#9b59b6', fg='white',
                            padx=20, pady=10, cursor='hand2')
        grid_btn.pack(side=tk.LEFT, padx=5)
        
        # 右侧 - 预测结果区域
        right_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # 结果标题
        result_title = tk.Label(right_frame, text="预测结果", 
                               font=("Arial", 14, "bold"), bg='white')
        result_title.pack(pady=8)
        
        # 预测的数字（大字体显示）
        self.predicted_label = tk.Label(right_frame, text="—", 
                                       font=("Arial", 60, "bold"), 
                                       fg='#4ecdc4', bg='white')
        self.predicted_label.pack(pady=10)
        
        # 置信度
        self.confidence_label = tk.Label(right_frame, text="置信度: —", 
                                         font=("Arial", 11), bg='white')
        self.confidence_label.pack()
        
        # 分隔线
        separator = ttk.Separator(right_frame, orient='horizontal')
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # 概率分布标题
        prob_title = tk.Label(right_frame, text="概率分布", 
                             font=("Arial", 11, "bold"), bg='white')
        prob_title.pack(pady=3)
        
        # 概率条框架
        self.prob_frame = tk.Frame(right_frame, bg='white')
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # 创建10个数字的概率条
        self.prob_bars = []
        self.prob_labels = []
        
        for i in range(10):
            # 数字标签
            digit_frame = tk.Frame(self.prob_frame, bg='white')
            digit_frame.pack(fill=tk.X, pady=2)
            
            digit_label = tk.Label(digit_frame, text=f"{i}:", 
                                  font=("Arial", 9, "bold"), 
                                  bg='white', width=2)
            digit_label.pack(side=tk.LEFT)
            
            # 进度条背景
            bar_bg = tk.Frame(digit_frame, bg='#e0e0e0', height=18)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # 进度条
            bar = tk.Frame(bar_bg, bg='#4ecdc4', height=18)
            bar.place(x=0, y=0, relheight=1, relwidth=0)
            self.prob_bars.append(bar)
            
            # 概率标签
            prob_label = tk.Label(digit_frame, text="0.0%", 
                                 font=("Arial", 8), bg='white', width=6)
            prob_label.pack(side=tk.LEFT)
            self.prob_labels.append(prob_label)
        
        # 模型信息
        model_info = tk.Label(right_frame, 
                             text=f"模型: {self.model_type.upper()}", 
                             font=("Arial", 8), bg='white', fg='gray')
        model_info.pack(side=tk.BOTTOM, pady=3)
        
    def start_draw(self, event):
        """开始绘制"""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_on_canvas(self, event):
        """在画布上绘制"""
        if self.is_drawing:
            x = max(0, min(event.x, self.display_size - 1))
            y = max(0, min(event.y, self.display_size - 1))
            
            # 将坐标转换到28x28的实际图像上
            img_x = x // self.scale_factor
            img_y = y // self.scale_factor
            last_img_x = self.last_x // self.scale_factor
            last_img_y = self.last_y // self.scale_factor
            
            # 在28x28的PIL图像上绘制
            self.draw.line([last_img_x, last_img_y, img_x, img_y],
                          fill='white', width=2)
            self.draw.ellipse([img_x-1, img_y-1, img_x+1, img_y+1], fill='white')
            
            # 在画布上绘制对应的放大方块
            self.redraw_canvas()
            
            self.last_x = x
            self.last_y = y
    
    def redraw_canvas(self):
        """根据28x28图像重绘画布（放大显示）"""
        self.canvas.delete('pixel')  # 删除之前的像素
        img_array = np.array(self.image)
        
        # 遍历28x28的每个像素，绘制对应的放大方块
        for i in range(self.mnist_size):
            for j in range(self.mnist_size):
                if img_array[j, i] > 0:  # 如果该像素被绘制
                    # 计算在280x280画布上的位置
                    x1 = i * self.scale_factor
                    y1 = j * self.scale_factor
                    x2 = x1 + self.scale_factor
                    y2 = y1 + self.scale_factor
                    
                    # 根据像素值设置灰度
                    gray_value = int(img_array[j, i])
                    color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
                    
                    # 绘制方块
                    self.canvas.create_rectangle(x1, y1, x2, y2, 
                                                fill=color, outline='',
                                                tags='pixel')
        
        # 确保网格在最上层
        if self.show_grid:
            self.canvas.tag_raise('grid')
    
    def stop_draw(self, event):
        """停止绘制"""
        self.is_drawing = False
        # 确保最后一次绘制完成
        self.redraw_canvas()
        # 自动识别
        self.predict()
        
    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.mnist_size, self.mnist_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # 重新绘制网格
        if self.show_grid:
            self.draw_grid()
        
        # 重置显示
        self.predicted_label.config(text="—")
        self.confidence_label.config(text="置信度: —")
        for i in range(10):
            self.prob_bars[i].place(relwidth=0)
            self.prob_labels[i].config(text="0.0%")
    
    def preprocess_image(self):
        """预处理图像为模型输入格式"""
        # MNIST格式：黑色背景(0)，白色数字(255)
        # 我们的画布也是黑底白字，所以直接使用，不需要反转
        img = self.image
        
        # 图像已经是28x28，无需缩放
        # 转换为numpy数组并归一化到[0,1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 应用与训练时相同的标准化：Normalize(mean=0.1307, std=0.3081)
        img_array = (img_array - 0.1307) / 0.3081
        
        # 根据模型类型reshape
        if self.model_type == 'cnn':
            img_array = img_array.reshape(1, 1, 28, 28)
        else:
            img_array = img_array.reshape(1, 784)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img_array).to(self.device)
        
        return img_tensor
    
    def save_image(self):
        """保存绘制的图像（用于调试）"""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存绘制图像（28x28，黑底白字，与MNIST格式一致）
            save_path = f"debug_mnist_{timestamp}.png"
            self.image.save(save_path)
            
            print(f"✓ 图像已保存: {save_path}")
            print(f"  格式: 28x28, 黑底白字(MNIST格式)")
        except Exception as e:
            print(f"保存失败: {e}")
    
    def predict(self):
        """执行预测"""
        try:
            # 预处理图像
            input_tensor = self.preprocess_image()
            
            # 预测
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_digit = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_digit].item()
            
            # 更新显示
            self.predicted_label.config(text=str(predicted_digit))
            self.confidence_label.config(text=f"置信度: {confidence*100:.1f}%")
            
            # 更新概率条
            probs = probabilities.cpu().numpy()
            for i in range(10):
                prob = probs[i]
                self.prob_bars[i].place(relwidth=prob)
                self.prob_labels[i].config(text=f"{prob*100:.1f}%")
                
                # 高亮预测的数字
                if i == predicted_digit:
                    self.prob_bars[i].config(bg='#ff6b6b')
                else:
                    self.prob_bars[i].config(bg='#4ecdc4')
            
        except Exception as e:
            print(f"预测失败: {e}")


def main():
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

