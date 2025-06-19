# -----------------------------------------------------------------------------
# 数据集生成脚本: 高斯梯度压力 (create_dataset_gaussian_gradient.py)
# 
# 功能:
# 1. 生成指定数量的样本。
# 2. 对每个样本，通过二维高斯函数生成一个中心强、四周弱的平滑电导率分布。
# 3. 对每个生成的电导率分布图(Y)，调用有限元求解器计算出对应的15维电阻向量(X)。
# 4. 将所有生成的 (X, Y) 数据对，分别保存到 X_data.npy 和 Y_data.npy 文件中。
#
# 使用方法:
# 1. 将此脚本放置在 'dataset/gaussian_gradient/' 目录下。
# 2. 确保 'structured_mesh_grid_new/' 文件夹与 'gaussian_gradient' 文件夹在同一级目录下。
# 3. 运行此脚本: python create_dataset_gaussian_gradient.py
# -----------------------------------------------------------------------------

# 导入必要的库
import numpy as np
import meshio
import os
import tqdm  # 引入 tqdm 来显示进度条

# 显式地从 skfem 的不同模块中导入所有需要的类和函数
from skfem import (MeshHex, ElementHex1, Basis)
from skfem.assembly import asm, BilinearForm
# 针对 v11.0.0, solve 位于 utils
from skfem.utils import solve
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve

# =============================================================================
# --- 1. 超参数定义 ---
# =============================================================================

# a) 数据集相关参数
OUTPUT_DIR = "."  # 将输出目录设为当前文件夹
TOTAL_SAMPLES_TO_GENERATE = 1000 # 您期望生成的样本总数

# 【新】定义高斯分布的参数范围
GAUSSIAN_RADIUS_RANGE = (8.0, 20.0) # 高斯函数的影响半径 (标准差)
PEAK_CONDUCTIVITY_RATIO_RANGE = (5.0, 25.0) # 压力中心的峰值电导率是背景的5到25倍

# b) 物理模型和有限元模型参数 (与之前保持一致)
BASE_CONDUCTIVITY = 1.0
MESH_DIR = "../structured_mesh_grid_new"
MESH_FILENAME = "material_mesh_3d.msh"
NODES_FILENAME = "electrode_nodes.npz"
VOLTAGE_DIFFERENCE = 1.0
MEASUREMENT_PAIRS = [
    ('top_1', 'bottom_1'), ('top_1', 'bottom_2'), ('top_1', 'bottom_3'),
    ('top_2', 'bottom_1'), ('top_2', 'bottom_2'), ('top_2', 'bottom_3'),
    ('top_3', 'bottom_1'), ('top_3', 'bottom_2'), ('top_3', 'bottom_3'),
    ('top_1', 'top_2'), ('top_2', 'top_3'), ('top_1', 'top_3'),
    ('bottom_1', 'bottom_2'), ('bottom_2', 'bottom_3'), ('bottom_1', 'bottom_3'),
]

# d) 网格几何参数 (必须与网格生成脚本完全一致)
NX, NY, NZ = 12, 12, 1
TOTAL_ELEMENTS = NX * NY * NZ

# =============================================================================
# --- 2. 核心函数定义 ---
# =============================================================================

def create_gaussian_gradient_map(element_midpoints):
    """
    通过二维高斯函数，生成一个具有平滑梯度的高电导率区域的电导率图。
    
    Args:
        element_midpoints (numpy.ndarray): 一个 (144, 3) 的数组，包含所有单元的中心点坐标。

    Returns:
        numpy.ndarray: 一个144维的向量，代表每个微元的电导率。
    """
    # a. 初始化背景电导率
    sigma_bg = BASE_CONDUCTIVITY + np.random.uniform(-0.05, 0.05)
    
    # b. 随机决定高斯函数的参数
    #   i. 随机中心点 (只在x,y平面上)
    center_x = np.random.uniform(0, NX * 5.0) 
    center_y = np.random.uniform(0, NY * 5.0)
    #   ii. 随机影响半径 (标准差)
    radius = np.random.uniform(*GAUSSIAN_RADIUS_RANGE)
    #   iii. 随机峰值电导率
    peak_ratio = np.random.uniform(*PEAK_CONDUCTIVITY_RATIO_RANGE)
    sigma_peak = sigma_bg * peak_ratio
    
    # c. 计算所有单元中心点到压力中心的距离的平方 (矢量化操作，非常高效)
    #    我们只关心x,y平面上的距离
    dist_sq = (element_midpoints[:, 0] - center_x)**2 + (element_midpoints[:, 1] - center_y)**2
    
    # d. 根据高斯函数计算每个单元的电导率增量
    gaussian_values = np.exp(-dist_sq / (2 * radius**2))
    
    # e. 计算最终的电导率图
    conductivity_map = sigma_bg + (sigma_peak - sigma_bg) * gaussian_values
    
    return conductivity_map


def solve_fem_for_map(stiffness_form, basis, electrode_nodes):
    """
    对于一个给定的电导率分布，计算出其对应的15维电阻测量向量。
    """
    A = asm(stiffness_form, basis)
    
    resistance_vector = []
    for drive_tag, ground_tag in MEASUREMENT_PAIRS:
        drive_dofs = electrode_nodes[drive_tag]
        ground_dofs = electrode_nodes[ground_tag]
        dirichlet_dofs = np.unique(np.concatenate([drive_dofs, ground_dofs]))
        
        active_dofs = basis.complement_dofs(dirichlet_dofs)
        phi = np.zeros(basis.N)
        phi[drive_dofs] = VOLTAGE_DIFFERENCE
        phi[ground_dofs] = 0.0
        
        A_free = A[active_dofs, :][:, active_dofs]
        b_free = -A[active_dofs, :][:, dirichlet_dofs] @ phi[dirichlet_dofs]
        
        phi_free = solve(A_free, b_free, solver=spsolve)
        phi[active_dofs] = phi_free
        
        full_force_vector = A @ phi
        total_current = np.sum(full_force_vector[ground_dofs])
        resistance = VOLTAGE_DIFFERENCE / abs(total_current)
        
        resistance_vector.append(resistance)
        
    return np.array(resistance_vector)


# =============================================================================
# --- 3. 主执行流程 ---
# =============================================================================

if __name__ == "__main__":
    
    # --- a. 加载通用文件 ---
    mesh_path = os.path.join(MESH_DIR, MESH_FILENAME)
    nodes_path = os.path.join(MESH_DIR, NODES_FILENAME)
    try:
        m = meshio.read(mesh_path)
        mesh = MeshHex(m.points.T, m.cells_dict['hexahedron'].T)
        electrode_nodes = np.load(nodes_path)
    except Exception as e:
        print(f"错误：无法加载所需文件。请确保路径 '{MESH_DIR}' 正确。错误信息: {e}")
        exit()

    # --- b. 初始化基函数和预计算 ---
    element = ElementHex1()
    basis = Basis(mesh, element)
    # 【重要】为了效率，在循环外一次性计算好所有单元的中心点
    element_midpoints = np.mean(mesh.p[:, mesh.t], axis=1).T
    
    # --- c. 准备数据存储 ---
    X_data_list = []
    Y_data_list = []
    
    # --- d. 主循环，生成数据 ---
    print(f"开始生成 {TOTAL_SAMPLES_TO_GENERATE} 个高斯梯度压力样本...")
    for i in tqdm.trange(TOTAL_SAMPLES_TO_GENERATE, desc="生成进度"):
        # i. 生成一个随机的电导率图 (Y)
        conductivity_map = create_gaussian_gradient_map(element_midpoints)
        
        # ii. 定义一个与此图绑定的刚度矩阵“配方”
        @BilinearForm
        def stiffness(u, v, w):
            sigma = conductivity_map[w.idx]
            return sigma * dot(grad(u), grad(v))
        
        # iii. 调用求解器，计算对应的电阻向量 (X)
        resistance_vector = solve_fem_for_map(stiffness, basis, electrode_nodes)
        
        # iv. 将这对数据 (X, Y) 追加到列表中
        X_data_list.append(resistance_vector)
        Y_data_list.append(conductivity_map)

    # --- e. 保存数据集 ---
    print("\n所有数据生成完毕，正在保存...")
    
    X_data = np.array(X_data_list)
    Y_data = np.array(Y_data_list)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    x_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_path = os.path.join(OUTPUT_DIR, "Y_data.npy")
    np.save(x_path, X_data)
    np.save(y_path, Y_data)
    
    print(f"数据集保存成功！")
    print(f"  - 输入数据 (X): {X_data.shape} -> 已保存至 {x_path}")
    print(f"  - 标签数据 (Y): {Y_data.shape} -> 已保存至 {y_path}")
