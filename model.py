# -----------------------------------------------------------------------------
# 神经网络模型定义 (model.py - 文献复现版)
#
# 功能:
# 1. 使用 PyTorch 定义一个与文献描述完全一致的全连接网络 (FCN)。
# 2. 结构: 输入层 -> 隐藏层1(1024, ReLU) -> 隐藏层2(1024, ReLU) -> Dropout(0.2) -> 输出层
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNModel(nn.Module):
    """
    一个复现自文献的全连接神经网络模型。
    """
    def __init__(self, input_dim=15, output_dim=144, dropout_rate=0.2):
        """
        在构造函数中定义模型的各个层。
        
        Args:
            input_dim (int): 输入向量的维度。
            output_dim (int): 输出向量的维度。
            dropout_rate (float): Dropout层的丢弃比率。
        """
        super(FCNModel, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, 1024)   # 第一个隐藏层
        self.fc2 = nn.Linear(1024, 1024)  # 第二个隐藏层
        self.dropout = nn.Dropout(p=dropout_rate) # Dropout层
        self.fc3 = nn.Linear(1024, output_dim)    # 输出层

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
        
        # 在进入输出层前，应用 Dropout
        x = self.dropout(x)
        
        # 输出层不使用激活函数 (线性输出)
        x = self.fc3(x)
        
        return x

if __name__ == '__main__':
    # 这个部分可以用来快速测试模型结构是否正确
    # 创建一个模型实例
    model = FCNModel()
    
    # 打印模型结构
    print("成功构建文献复现的 FCN 模型，结构如下：")
    print(model)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params}")

