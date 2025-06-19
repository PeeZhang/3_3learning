# -----------------------------------------------------------------------------
# 模型训练脚本 (train.py - PyTorch版)
#
# 功能:
# 1. 加载所有子文件夹中的数据集 (X_data.npy, Y_data.npy)。
# 2. 将数据转换为 PyTorch Tensor，并创建 DataLoader。
# 3. 为训练数据实时添加随机噪声。
# 4. 初始化我们在 model.py 中定义的 PyTorch MLP 模型。
# 5. 编写训练和验证循环，训练模型并记录损失(loss)变化。
# 6. 将训练好的模型权重保存到文件。
# 7. 绘制训练过程的损失曲线图。
#
# 使用方法:
# 1. 将此脚本放置在项目主文件夹中。
# 2. 确保 'dataset/' 文件夹及其子文件夹结构正确。
# 3. 安装所需库: pip install torch matplotlib
# 4. 运行此脚本: python train.py
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 从我们自己编写的 model.py 文件中导入模型类
from model import MLPModel

# =============================================================================
# --- 1. 超参数定义 ---
# =============================================================================

# a) 数据集相关参数
DATASET_BASE_DIR = "dataset"
DATASET_CATEGORIES = ["single_point", "multi_point", "random_pixels", "large_area", "gaussian_gradient"]

# b) 训练相关参数
EPOCHS = 10000
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
NOISE_LEVEL = 0.01

# =============================================================================
# --- 2. 数据加载函数 (与之前相同) ---
# =============================================================================
def load_all_data(base_dir, categories):
    all_x, all_y = [], []
    print("开始加载数据集...")
    for category in categories:
        x_path = os.path.join(base_dir, category, "X_data.npy")
        y_path = os.path.join(base_dir, category, "Y_data.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"  - 正在加载 '{category}' 数据...")
            all_x.append(np.load(x_path))
            all_y.append(np.load(y_path))
        else:
            print(f"  - 警告：在 '{category}' 文件夹下未找到数据，已跳过。")
    if not all_x: return None, None
    return np.vstack(all_x), np.vstack(all_y)

# =============================================================================
# --- 3. 主执行流程 ---
# =============================================================================
if __name__ == "__main__":
    
    # --- a. 加载并准备数据 ---
    X_data, Y_data = load_all_data(DATASET_BASE_DIR, DATASET_CATEGORIES)
    if X_data is None: exit()
        
    print("\n数据集加载完毕！")
    print(f"  - 总输入数据 (X) 形状: {X_data.shape}")
    print(f"  - 总标签数据 (Y) 形状: {Y_data.shape}")

    # 【核心】为输入数据增加噪声
    print(f"为输入数据增加 {NOISE_LEVEL*100}% 的噪声...")
    noise = np.random.normal(1.0, NOISE_LEVEL, X_data.shape)
    X_data_noisy = X_data * noise
    
    # 将 NumPy 数组转换为 PyTorch Tensor
    X_tensor = torch.tensor(X_data_noisy, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

    # 创建 PyTorch 数据集
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    # 划分训练集和验证集
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器 (DataLoaders)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- b. 初始化模型、损失函数和优化器 ---
    print("\n正在初始化 PyTorch MLP 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"将使用 '{device}' 设备进行训练。")
    
    model = MLPModel().to(device)
    loss_fn = nn.MSELoss() # 使用均方误差作为损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- c. 训练模型 ---
    print("\n开始训练模型...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train() # 将模型设置为训练模式
        train_loss_epoch = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    # 清除旧的梯度
            outputs = model(inputs)  # 前向传播
            loss = loss_fn(outputs, labels) # 计算损失
            loss.backward()          # 反向传播
            optimizer.step()         # 更新权重
            
            train_loss_epoch += loss.item()

        # 计算并记录平均训练损失
        avg_train_loss = train_loss_epoch / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval() # 将模型设置为评估模式
        val_loss_epoch = 0.0
        with torch.no_grad(): # 在验证时，不计算梯度
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss_epoch += loss.item()
        
        # 计算并记录平均验证损失
        avg_val_loss = val_loss_epoch / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"周期 [{epoch+1}/{EPOCHS}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

    print("模型训练完成！")

    # --- d. 保存模型 ---
    torch.save(model.state_dict(), "pytorch_mlp_model.pth")
    print("\n训练好的模型权重已保存至 'pytorch_mlp_model.pth'")

    # --- e. 可视化训练历史 ---
    print("正在绘制训练历史曲线...")
    plt.figure(figsize=(10, 6))
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("\n警告：未找到 'SimHei' 字体，图表标题可能显示为方框。")

    plt.plot(history['train_loss'], label='训练损失 (Training Loss)')
    plt.plot(history['val_loss'], label='验证损失 (Validation Loss)')
    plt.title('模型损失变化曲线 (PyTorch)')
    plt.xlabel('训练周期 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("pytorch_training_history.png")
    print("训练历史曲线图已保存至 'pytorch_training_history.png'")
    plt.show()

