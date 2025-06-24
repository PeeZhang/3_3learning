# -----------------------------------------------------------------------------
# 模型预测结果可视化与评估脚本 (visualize_predictions.py)
#
# 【PyTorch 版本 - v2 多数据集加载】
# 功能:
# 1. 加载预训练好的PyTorch模型。
# 2. 根据 DATASET_CATEGORIES 参数，加载一个或多个测试数据集并合并。
# 3. 对合并后的测试数据进行预测。
# 4. 可视化对比“真实电导率图”、“预测电导率图”和“绝对误差图”。
# 5. 计算并显示每个预测结果的相对图像误差(RIE)和图像相关系数(ICC)。
#
# 使用方法:
# 1. 将此脚本与您的 model.py 文件放置在同一项目目录下。
# 2. 修改 DATASET_CATEGORIES 列表来选择要加载的数据集。
# 3. 确保模型和数据集路径正确。
# 4. 运行此脚本: python visualize_predictions.py
# -----------------------------------------------------------------------------

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch

# 从您的模型定义文件 model.py 中导入 FCNModel 类
try:
    from model import FCNModel
except ImportError:
    print("错误: 无法导入 FCNModel。请确保 'model.py' 文件与此脚本在同一目录下。")
    exit()


# =============================================================================
# --- 1. 参数和文件路径定义 ---
# =============================================================================
# 模型路径
MODEL_PATH = './single_multi_gaussian_baseline/fcn_model_replicated.pth' 

# 【新】定义要加载的数据集类别
DATASET_BASE_DIR = "./dataset/"
DATASET_CATEGORIES = ["baseline", "single_point", "multi_point", "gaussian_gradient"] # 您可以选择一个或多个类别

# 可视化参数
NUM_SAMPLES_TO_VISUALIZE = 5  # 您希望可视化多少个样本
NX, NY = 12, 12               # 网格尺寸，必须与生成数据时一致

# =============================================================================
# --- 2. 核心函数定义 (与之前相同) ---
# =============================================================================

def calculate_rie(y_true, y_pred):
    """
    计算相对图像误差 (Relative Image Error)。
    """
    numerator = np.linalg.norm(y_pred - y_true)
    denominator = np.linalg.norm(y_true)
    if denominator == 0:
        return 0
    return numerator / denominator

def calculate_icc(y_true, y_pred):
    """
    计算图像相关系数 (Image Correlation Coefficient)。
    """
    true_flat = y_true.flatten()
    pred_flat = y_pred.flatten()
    corr_matrix = np.corrcoef(true_flat, pred_flat)
    return corr_matrix[0, 1]

def visualize_and_evaluate(model, x_test, y_test, num_samples, device):
    """
    随机选择样本，进行预测、评估和可视化。
    """
    num_available_samples = x_test.shape[0]
    if num_available_samples < num_samples:
        print(f"警告: 可用样本数量 ({num_available_samples}) 小于期望可视化的数量 ({num_samples})。将使用所有可用样本。")
        num_samples = num_available_samples
        
    sample_indices = random.sample(range(num_available_samples), num_samples)

    for i, index in enumerate(sample_indices):
        # a. 准备数据
        x_sample_np = x_test[index]
        y_true = y_test[index].reshape(NY, NX)
        
        # --- b. PyTorch 模型预测 ---
        x_sample_tensor = torch.from_numpy(x_sample_np).float().to(device)
        x_sample_batch = x_sample_tensor.unsqueeze(0)
        
        with torch.no_grad():
            y_pred_tensor = model(x_sample_batch)
        
        y_pred_flat = y_pred_tensor.cpu().numpy()[0]
        y_pred = y_pred_flat.reshape(NY, NX)
        
        # c. 计算评估指标
        rie = calculate_rie(y_true, y_pred)
        icc = calculate_icc(y_true, y_pred)
        
        # d. 可视化
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'样本 #{index} 的预测结果\nRIE = {rie:.4f} | ICC = {icc:.4f}', fontsize=16)

        # 子图1: 真实电导率图
        im1 = axes[0].imshow(y_true, cmap='viridis', origin='lower')
        axes[0].set_title('真实电导率图 (Ground Truth)')
        axes[0].set_xlabel('X 单元索引')
        axes[0].set_ylabel('Y 单元索引')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # 子图2: 预测电导率图
        im2 = axes[1].imshow(y_pred, cmap='viridis', origin='lower', vmin=y_true.min(), vmax=y_true.max())
        axes[1].set_title('预测电导率图 (Prediction)')
        axes[1].set_xlabel('X 单元索引')
        axes[1].set_ylabel('')
        fig.colorbar(im2, ax=axes[0], fraction=0.046, pad=0.04) # 复用第一个colorbar的设置

        # 子图3: 绝对误差图
        diff = np.abs(y_pred - y_true)
        im3 = axes[2].imshow(diff, cmap='magma', origin='lower')
        axes[2].set_title('绝对误差图 (Absolute Error)')
        axes[2].set_xlabel('X 单元索引')
        axes[2].set_ylabel('')
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(f"prediction_sample_{index}.png")
        print(f"已保存样本 #{index} 的可视化结果至 prediction_sample_{index}.png")
        plt.show()

# =============================================================================
# --- 3. 主执行流程 ---
# =============================================================================

if __name__ == "__main__":
    
    # --- a. 设置设备 (优先使用GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")
    
    # --- b. 加载模型 ---
    print(f"正在从 '{MODEL_PATH}' 加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 '{MODEL_PATH}'。请检查路径是否正确。")
        exit()
    try:
        model = FCNModel() 
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() 
        print(model)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        exit()
    print("模型加载成功！")
    
    # --- c. 【新】加载并合并多个类别的数据集 ---
    X_test_list = []
    Y_test_list = []
    print("\n开始加载测试数据...")
    for category in DATASET_CATEGORIES:
        print(f"  > 正在加载类别: {category}")
        x_path = os.path.join(DATASET_BASE_DIR, category, 'X_data.npy')
        y_path = os.path.join(DATASET_BASE_DIR, category, 'Y_data.npy')
        
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            print(f"    - 警告: 在 '{os.path.join(DATASET_BASE_DIR, category)}' 目录下找不到数据文件，已跳过。")
            continue
        try:
            X_test_list.append(np.load(x_path))
            Y_test_list.append(np.load(y_path))
            print(f"    - 成功加载 {len(X_test_list[-1])} 条数据。")
        except Exception as e:
            print(f"    - 加载数据时出错: {e}")
            continue

    if not X_test_list:
        print("\n错误: 未能加载任何测试数据。请检查 DATASET_CATEGORIES 和路径设置。")
        exit()
        
    # 合并所有加载的数据
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)
    print(f"\n所有数据加载完毕！总测试样本数: {len(X_test)}")
    
    # --- d. 执行可视化和评估 ---
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        print("\n警告：未找到 'SimHei' 字体，图表标题可能显示为方框。")
        
    visualize_and_evaluate(model, X_test, Y_test, NUM_SAMPLES_TO_VISUALIZE, device)

