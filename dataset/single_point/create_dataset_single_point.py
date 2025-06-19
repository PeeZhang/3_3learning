# -----------------------------------------------------------------------------
# 数据集生成脚本: 单点压力 (create_dataset_single_point.py)
# 
# 功能:
# 1. 按照预设的形状列表，为每种形状生成指定数量的样本。
# 2. 对每个样本，随机生成受压区域的位置和电导率幅值。
# 3. 对每个生成的电导率分布图(Y)，调用有限元求解器计算出对应的15维电阻向量(X)。
# 4. 将所有生成的 (X, Y) 数据对，分别保存到 X_data.npy 和 Y_data.npy 文件中。
#
# 使用方法:
# 1. 将此脚本放置在 'dataset/single_point/' 目录下。
# 2. 确保 'structured_mesh_grid_new/' 文件夹与 'single_point' 文件夹在同一级目录下。
# 3. 运行此脚本: python create_dataset_single_point.py
# -----------------------------------------------------------------------------

# 导入必要的库
import numpy as np
import meshio
import os
import time
import tqdm  # 引入 tqdm 来显示进度条

# 显式地从 skfem 的不同模块中导入所有需要的类和函数
from skfem import (MeshHex, ElementHex1, Basis)
from skfem.assembly import asm, BilinearForm
# 针对 v11.0.0, solve 位于 utils
from skfem.utils import solve
from skfem.helpers import dot, grad
# 【重要修正】导入 SciPy 的稀疏求解器
from scipy.sparse.linalg import spsolve

# =============================================================================
# --- 1. 超参数定义 ---
# =============================================================================

# a) 数据集相关参数
OUTPUT_DIR = "."  # 将输出目录设为当前文件夹
# 定义要生成的受压区域的形状 (height, width)
# 对于单点压力，我们可以定义不同大小的“点”
SHAPES_TO_GENERATE = [(1, 1), (2, 2), (3, 3), (1,2), (2,1), (2,3), (3,2)] 
SAMPLES_PER_SHAPE = 200  # 为每一种形状生成10个样本 (为了快速测试，可以设小一点)

# b) 物理模型相关参数
# 定义电导率范围
BASE_CONDUCTIVITY = 1.0  # 未受压区域的基准电导率
CONDUCTIVITY_RATIO_RANGE = (2.0, 20.0) # 受压区域电导率是背景的2到20倍

# c) 有限元模型相关参数
# 根据您的说明，更新为正确的相对路径
MESH_DIR = "../structured_mesh_grid_new" # 网格文件所在的相对路径
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

# =============================================================================
# --- 2. 核心函数定义 ---
# =============================================================================

def create_random_conductivity_map(shape):
    """
    根据给定的形状，生成一个随机的电导率分布图。
    
    Args:
        shape (tuple): 一个元组，如 (height, width)，定义了受压区域的尺寸。
    
    Returns:
        numpy.ndarray: 一个144维的向量，代表每个微元的电导率。
    """
    # a) 初始化电导率图
    # 增加一点随机扰动，提升模型鲁棒性
    sigma_bg = BASE_CONDUCTIVITY + np.random.uniform(-0.05, 0.05)
    conductivity_map = np.full(NX * NY * NZ, sigma_bg)
    
    # b) 随机确定电导率幅值
    ratio = np.random.uniform(*CONDUCTIVITY_RATIO_RANGE)
    sigma_pressed = sigma_bg * ratio
    
    # c) 随机确定受压位置
    shape_height, shape_width = shape
    max_x = NX - shape_width
    max_y = NY - shape_height
    # 随机生成矩形区域左下角的单元坐标 (i, j)
    x_loc = np.random.randint(0, max_x + 1)
    y_loc = np.random.randint(0, max_y + 1)
    
    # d) 将高电导率值赋给对应的微元
    for j in range(y_loc, y_loc + shape_height):
        for i in range(x_loc, x_loc + shape_width):
            # 我们假设压力只在最顶层 (k = NZ - 1)
            k = NZ - 1
            element_index = k * NY * NX + j * NX + i
            conductivity_map[element_index] = sigma_pressed
            
    return conductivity_map


def solve_fem_for_map(stiffness_form, basis, electrode_nodes):
    """
    对于一个给定的电导率分布 (已包含在 stiffness_form 中), 
    计算出其对应的15维电阻测量向量。
    
    Args:
        stiffness_form (skfem.BilinearForm): 已定义好的、包含电导率信息的刚度矩阵“配方”。
        basis (skfem.Basis): 有限元基函数。
        electrode_nodes (dict): 包含电极节点信息的字典。
        
    Returns:
        numpy.ndarray: 一个15维的电阻向量。
    """
    # a) 对于给定的电导率图，刚度矩阵A是固定不变的，只需计算一次
    A = asm(stiffness_form, basis)
    
    # b) 循环测量所有电极对
    resistance_vector = []
    for drive_tag, ground_tag in MEASUREMENT_PAIRS:
        # i. 定义当前测量对的边界条件
        drive_dofs = electrode_nodes[drive_tag]
        ground_dofs = electrode_nodes[ground_tag]
        dirichlet_dofs = np.unique(np.concatenate([drive_dofs, ground_dofs]))
        
        # ii. 求解线性系统
        active_dofs = basis.complement_dofs(dirichlet_dofs)
        phi = np.zeros(basis.N)
        phi[drive_dofs] = VOLTAGE_DIFFERENCE
        phi[ground_dofs] = 0.0
        
        A_free = A[active_dofs, :][:, active_dofs]
        b_free = -A[active_dofs, :][:, dirichlet_dofs] @ phi[dirichlet_dofs]
        
        # 【重要修正】直接将 spsolve 函数对象作为参数传递
        phi_free = solve(A_free, b_free, solver=spsolve)
        phi[active_dofs] = phi_free
        
        # iii. 计算总电流和电阻
        full_force_vector = A @ phi
        total_current = np.sum(full_force_vector[ground_dofs])
        resistance = VOLTAGE_DIFFERENCE / abs(total_current)
        
        # iv. 将结果存入向量
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

    # --- b. 初始化基函数 ---
    element = ElementHex1()
    basis = Basis(mesh, element)
    
    # --- c. 准备数据存储 ---
    X_data_list = []
    Y_data_list = []
    
    # --- d. 主循环，生成数据 ---
    total_samples = 0
    for shape in SHAPES_TO_GENERATE:
        total_samples += SAMPLES_PER_SHAPE
    
    # 使用tqdm创建总进度条
    with tqdm.tqdm(total=total_samples, desc="总进度") as pbar_total:
        for shape in SHAPES_TO_GENERATE:
            for i in range(SAMPLES_PER_SHAPE):
                # i. 生成一个随机的电导率图 (Ground Truth Y)
                conductivity_map = create_random_conductivity_map(shape)
                
                # ii. 定义一个与此图绑定的刚度矩阵“配方”
                @BilinearForm
                def stiffness(u, v, w):
                    sigma = conductivity_map[w.idx]
                    return sigma * dot(grad(u), grad(v))
                
                # iii. 调用求解器，计算对应的电阻向量 (Input X)
                resistance_vector = solve_fem_for_map(stiffness, basis, electrode_nodes)
                
                # iv. 将这对数据 (X, Y) 追加到列表中
                X_data_list.append(resistance_vector)
                Y_data_list.append(conductivity_map)
                
                # 更新总进度条
                pbar_total.set_description(f"总进度 (形状 {shape})")
                pbar_total.update(1)

    # --- e. 保存数据集 ---
    print("\n所有数据生成完毕，正在保存...")
    
    # 将列表转换为 NumPy 数组
    X_data = np.array(X_data_list)
    Y_data = np.array(Y_data_list)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存文件
    x_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_path = os.path.join(OUTPUT_DIR, "Y_data.npy")
    np.save(x_path, X_data)
    np.save(y_path, Y_data)
    
    print(f"数据集保存成功！")
    print(f"  - 输入数据 (X): {X_data.shape} -> 已保存至 {x_path}")
    print(f"  - 标签数据 (Y): {Y_data.shape} -> 已保存至 {y_path}")

