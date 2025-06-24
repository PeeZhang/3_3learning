# -----------------------------------------------------------------------------
# 模型训练脚本 (train.py - 文献复现版)
#
# 功能:
# 1. 加载所有子文件夹中的数据集。
# 2. 为训练数据实时添加随机噪声。
# 3. 初始化并训练我们在 model.py 中定义的 FCN 模型。
# 4. 使用文献中指定的超参数进行训练。
# 5. 保存模型并可视化训练过程。
#
# 使用方法:
# 1. 确保已在 conda 虚拟环境中安装了所有依赖库。
# 2. 在 Anaconda Prompt 中，激活环境 (`conda activate pytorch-fem`)。
# 3. 导航到此脚本所在的目录。
# 4. 运行此脚本: python train.py
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 从我们自己编写的 model.py 文件中导入新的 FCN 模型类
from model import FCNModel

# =============================================================================
# --- 1. 超参数定义 (根据文献更新) ---
# =============================================================================

# a) 数据集相关参数
DATASET_BASE_DIR = "dataset"
# DATASET_CATEGORIES = ["gaussian_gradient"]
# DATASET_CATEGORIES = ["single_point"]
# DATASET_CATEGORIES = ["multi_point"]
# DATASET_CATEGORIES = ["large_area"]
# DATASET_CATEGORIES = ["gaussian_gradient", "baseline"]
# DATASET_CATEGORIES = ["single_point", "baseline"]
# DATASET_CATEGORIES = ["single_point", "gaussian_gradient", "baseline"]
DATASET_CATEGORIES = ["single_point", "multi_point", "baseline", "gaussian_gradient"]
# DATASET_CATEGORIES = ["single_point", "multi_point", "random_pixels", "large_area", "gaussian_gradient", "baseline"]

# b) 训练相关参数
EPOCHS = 5000
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.01
NOISE_LEVEL = 0.01

# =============================================================================
# --- 2. 数据加载函数 (保持不变) ---
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

    # 为输入数据增加噪声
    print(f"为输入数据增加 {NOISE_LEVEL*100}% 的噪声...")
    noise = np.random.normal(1.0, NOISE_LEVEL, X_data.shape)
    X_data_noisy = X_data * noise
    
    # 转换为 PyTorch Tensor
    X_tensor = torch.tensor(X_data_noisy, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_data, dtype=torch.float32)

    # 创建并划分数据集
    dataset = TensorDataset(X_tensor, Y_tensor)
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器 (DataLoaders)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- b. 初始化模型、损失函数和优化器 ---
    print("\n正在初始化文献复现的 FCN 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"将使用 '{device}' 设备进行训练。")
    
    model = FCNModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- c. 训练模型 ---
    print(f"\n开始训练模型 ({EPOCHS} 个周期)...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss_epoch = 0.0
        
        # 使用 tqdm 显示训练进度条
        train_loop = tqdm(train_loader, desc=f"周期 {epoch+1}/{EPOCHS} [训练]", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_epoch / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss_epoch += loss.item()
        
        avg_val_loss = val_loss_epoch / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0: # 每10个周期打印一次日志
            print(f"周期 [{epoch+1}/{EPOCHS}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

    print("模型训练完成！")

    # --- d. 保存模型 ---
    torch.save(model.state_dict(), "fcn_model_replicated.pth")
    print("\n训练好的模型权重已保存至 'fcn_model_replicated.pth'")

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
    plt.title('FCN 模型损失变化曲线')
    plt.xlabel('训练周期 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("fcn_training_history.png")
    print("训练历史曲线图已保存至 'fcn_training_history.png'")
    plt.show()

