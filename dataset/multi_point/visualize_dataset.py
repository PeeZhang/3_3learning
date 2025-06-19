# -----------------------------------------------------------------------------
# 数据集对比可视化脚本 (visualize_dataset_comparison.py)
#
# 功能:
# 1. 加载生成的数据集文件 (X_data.npy, Y_data.npy)。
# 2. 随机挑选指定数量的样本。
# 3. 在同一个窗口中，为每个样本纵向绘制其电导率分布图和电阻测量向量。
# 4. 将所有样本横向排列以便对比。
# 5. 统一所有电导率图的颜色范围 (Colorbar)。
# 6. 统一所有电阻图的Y轴尺度。
#
# 使用方法:
# 1. 将此脚本放置在 'dataset/single_point/' 或 'dataset/multi_point/' 等目录下。
# 2. 确保 'X_data.npy' 和 'Y_data.npy' 文件在同一目录中。
# 3. 运行此脚本: python visualize_dataset_comparison.py
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# =============================================================================
# --- 1. 超参数定义 ---
# =============================================================================

# 定义要加载的数据文件名
X_DATA_FILENAME = "X_data.npy"
Y_DATA_FILENAME = "Y_data.npy"

# 定义要随机抽样并可视化的样本数量
NUM_SAMPLES_TO_SHOW = 3 # 建议一次不要显示太多，以免窗口过宽

# 定义网格的几何参数 (必须与生成脚本中的完全一致)
NX, NY = 12, 12

# =============================================================================
# --- 2. 主执行流程 ---
# =============================================================================

if __name__ == "__main__":
    
    # --- a. 加载数据集文件 ---
    try:
        X_data = np.load(X_DATA_FILENAME)
        Y_data = np.load(Y_DATA_FILENAME)
        print(f"成功加载数据集: ")
        print(f"  - 输入数据 (X) 形状: {X_data.shape}")
        print(f"  - 标签数据 (Y) 形状: {Y_data.shape}")
    except FileNotFoundError:
        print(f"错误：无法找到 '{X_DATA_FILENAME}' 或 '{Y_DATA_FILENAME}'。")
        print("请确保此脚本与数据集文件在同一个目录下，并已成功运行生成脚本。")
        exit()
        
    num_total_samples = X_data.shape[0]
    if num_total_samples < NUM_SAMPLES_TO_SHOW:
        print(f"警告：数据集中样本总数 ({num_total_samples}) 小于期望显示的数量 ({NUM_SAMPLES_TO_SHOW})。将显示所有样本。")
        samples_to_show = num_total_samples
    else:
        samples_to_show = NUM_SAMPLES_TO_SHOW

    # --- b. 随机选择样本 ---
    random_indices = np.random.choice(num_total_samples, size=samples_to_show, replace=False)
    
    print(f"\n将随机显示以下索引的 {samples_to_show} 个样本: {random_indices}")
    
    # --- c. 确定全局的可视化范围 ---
    selected_Y_data = Y_data[random_indices]
    global_vmin = np.min(selected_Y_data)
    global_vmax = np.max(selected_Y_data)
    
    selected_X_data = X_data[random_indices]
    global_ymax_resistance = np.max(selected_X_data) * 1.1 # 增加10%的边距

    # --- d. 【核心修改】创建横向排列的子图网格 ---
    # 我们创建 2 行，samples_to_show 列的子图
    fig, axes = plt.subplots(nrows=2, ncols=samples_to_show, figsize=(6 * samples_to_show, 10))
    fig.suptitle('数据集随机样本对比可视化', fontsize=18, y=0.98)

    # 如果只显示一个样本，axes不是二维数组，需要特殊处理
    if samples_to_show == 1:
        axes = axes[:, np.newaxis] # 将 (2,) 变为 (2, 1)

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 优先使用黑体
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    except Exception:
        print("\n警告：未找到 'SimHei' 字体，图表标题可能显示为方框。")

    # --- e. 循环绘图 ---
    for i, sample_index in enumerate(random_indices):
        
        conductivity_map_vector = Y_data[sample_index]
        resistance_vector = X_data[sample_index]
        
        conductivity_map_image = conductivity_map_vector.reshape(NY, NX)
        
        # --- 在第一行子图中绘制电导率分布图 (Y) ---
        ax1 = axes[0, i]
        ax1.set_title(f'样本 {sample_index}: 电导率分布图 (Y)')
        im = ax1.imshow(conductivity_map_image, cmap='viridis', origin='lower',
                        interpolation='nearest', vmin=global_vmin, vmax=global_vmax)
        ax1.set_xlabel('X 单元索引')
        ax1.set_ylabel('Y 单元索引')
        fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

        # --- 在第二行子图中绘制电阻测量向量 (X) ---
        ax2 = axes[1, i]
        ax2.set_title(f'样本 {sample_index}: 电阻测量向量 (X)')
        measurement_indices = np.arange(1, len(resistance_vector) + 1)
        # 保持减小的柱状图宽度
        ax2.bar(measurement_indices, resistance_vector, color='skyblue', width=0.6)
        ax2.set_xlabel('测量电极对索引 (1-15)')
        ax2.set_ylabel('电阻值 (Ω)')
        ax2.set_xticks(measurement_indices)
        ax2.tick_params(axis='x', labelrotation=90) # 将x轴标签旋转90度，防止重叠
        ax2.set_ylim(0, global_ymax_resistance)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 调整布局，增加垂直和水平间距以避免重叠
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=3.0)
    plt.show()

