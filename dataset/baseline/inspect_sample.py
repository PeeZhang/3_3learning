# -----------------------------------------------------------------------------
# 指定样本可视化与数据检查脚本 (inspect_sample.py)
#
# 功能:
# 1. 加载当前目录下的数据集文件 (X_data.npy, Y_data.npy)。
# 2. 根据用户在超参数中指定的索引列表，精确地提取、打印并可视化样本。
# 3. 在控制台输出指定样本的X和Y向量的具体数值。
# 4. 在同一个窗口中，纵向对比展示所有指定样本的电导率图和电阻向量。
#
# 使用方法:
# 1. 将此脚本放置在您想检查的数据集子文件夹中 (例如 'dataset/gaussian_gradient/')。
# 2. 修改下面的 INDICES_TO_INSPECT 列表，填入您想查看的样本索引。
# 3. 运行此脚本: python inspect_sample.py
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# --- 1. 超参数定义 ---
# =============================================================================

# 【重要】请在这里指定您想查看的一个或多个样本的索引
INDICES_TO_INSPECT = [0, 50, 99] 

# 定义要加载的数据文件名
X_DATA_FILENAME = "X_data.npy"
Y_DATA_FILENAME = "Y_data.npy"

# 定义网格的几何参数 (必须与生成脚本中的完全一致)
NX, NY = 12, 12

# =============================================================================
# --- 2. 主执行流程 ---
# =============================================================================

if __name__ == "__main__":
    
    # --- a. 加载数据集文件 ---
    if not os.path.exists(X_DATA_FILENAME) or not os.path.exists(Y_DATA_FILENAME):
        print(f"错误：无法找到 '{X_DATA_FILENAME}' 或 '{Y_DATA_FILENAME}'。")
        print("请确保此脚本与数据集文件在同一个目录下。")
        exit()
        
    X_data = np.load(X_DATA_FILENAME)
    Y_data = np.load(Y_DATA_FILENAME)
    print(f"成功加载数据集 (X shape: {X_data.shape}, Y shape: {Y_data.shape})")
    
    # 检查指定的索引是否有效
    valid_indices = []
    for idx in INDICES_TO_INSPECT:
        if idx >= len(X_data):
            print(f"警告：索引 {idx} 超出范围 (数据集总大小为 {len(X_data)})，已跳过。")
        else:
            valid_indices.append(idx)

    if not valid_indices:
        print("错误：没有提供任何有效的索引进行可视化。")
        exit()

    num_samples_to_show = len(valid_indices)
    
    # --- b. 打印指定样本的向量数据 ---
    print("\n" + "="*30)
    print("--- 指定样本的向量数据 ---")
    print("="*30)
    for i, sample_index in enumerate(valid_indices):
        resistance_vector = X_data[sample_index]
        conductivity_vector = Y_data[sample_index]
        
        print(f"\n--- 样本索引 (Sample Index): {sample_index} ---")
        # 使用 np.array2string 格式化输出，方便查看
        print("  - 输入 X (15维电阻向量):")
        print(f"    {np.array2string(resistance_vector, formatter={'float_kind':lambda x: '%.4f' % x})}")
        print("\n  - 标签 Y (144维电导率向量):")
        print(f"    {np.array2string(conductivity_vector, formatter={'float_kind':lambda x: '%.4f' % x}, max_line_width=120)}")
    print("\n" + "="*30)

    # --- c. 确定全局的可视化范围 ---
    selected_Y_data = Y_data[valid_indices]
    global_vmin = np.min(selected_Y_data)
    global_vmax = np.max(selected_Y_data)
    
    selected_X_data = X_data[valid_indices]
    global_ymax_resistance = np.max(selected_X_data) * 1.1 # 增加10%的边距

    # --- d. 创建并排对比的子图网格 ---
    fig, axes = plt.subplots(nrows=2, ncols=num_samples_to_show, figsize=(6 * num_samples_to_show, 10))
    fig.suptitle('指定样本数据可视化', fontsize=18, y=0.98)

    # 如果只显示一个样本，axes不是二维数组，需要特殊处理
    if num_samples_to_show == 1:
        axes = axes[:, np.newaxis] 

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 优先使用黑体
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    except Exception:
        print("\n警告：未找到 'SimHei' 字体，图表标题可能显示为方框。")

    # --- e. 循环绘图 ---
    for i, sample_index in enumerate(valid_indices):
        
        conductivity_map_vector = Y_data[sample_index]
        resistance_vector = X_data[sample_index]
        
        conductivity_map_image = conductivity_map_vector.reshape(NY, NX)
        
        # 在第一行子图中绘制电导率分布图 (Y)
        ax1 = axes[0, i]
        ax1.set_title(f'样本 {sample_index}: 电导率图 (Y)')
        im = ax1.imshow(conductivity_map_image, cmap='viridis', origin='lower',
                        interpolation='nearest', vmin=global_vmin, vmax=global_vmax)
        ax1.set_xlabel('X 单元索引')
        if i == 0: ax1.set_ylabel('Y 单元索引')
        fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

        # 在第二行子图中绘制电阻测量向量 (X)
        ax2 = axes[1, i]
        ax2.set_title(f'样本 {sample_index}: 电阻向量 (X)')
        measurement_indices = np.arange(1, len(resistance_vector) + 1)
        ax2.bar(measurement_indices, resistance_vector, color='skyblue', width=0.6)
        ax2.set_xlabel('测量电极对索引 (1-15)')
        if i == 0: ax2.set_ylabel('电阻值 (Ω)')
        ax2.set_xticks(measurement_indices)
        ax2.tick_params(axis='x', labelrotation=90) 
        ax2.set_ylim(0, global_ymax_resistance)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

