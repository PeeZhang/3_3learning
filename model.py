# -----------------------------------------------------------------------------
# 神经网络模型定义 (model.py - PyTorch版)
#
# 功能:
# 1. 使用 PyTorch 定义一个全连接神经网络 (MLP) 模型。
# 2. 模型的输入维度是15 (对应15个电阻测量值)。
# 3. 模型的输出维度是144 (对应12x12的电导率图)。
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    """
    一个继承自 torch.nn.Module 的全连接神经网络模型。
    """
    def __init__(self, input_dim=15, output_dim=144):
        """
        在构造函数中定义模型的各个层。
        
        Args:
            input_dim (int): 输入向量的维度。
            output_dim (int): 输出向量的维度。
        """
        super(MLPModel, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        # 输出层
        self.fc4 = nn.Linear(1024, output_dim)

    def forward(self, x):
        """
        定义数据通过网络的前向传播路径。
        
        Args:
            x (torch.Tensor): 输入的张量数据。
            
        Returns:
            torch.Tensor: 模型的输出。
        """
        # 通过隐藏层并应用 ReLU 激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 输出层不使用激活函数 (线性输出)
        x = self.fc4(x)
        
        return x

if __name__ == '__main__':
    # 这个部分可以用来快速测试模型结构是否正确
    # 创建一个模型实例
    model = MLPModel()
    
    # 打印模型结构
    print("成功构建 PyTorch MLP 模型，结构如下：")
    print(model)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params}")

