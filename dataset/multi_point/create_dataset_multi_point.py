# -----------------------------------------------------------------------------
# 数据集生成脚本: 多点压力 (create_dataset_multi_point.py)
# 
# 【v12 核心集成版】
# 功能:
# 1. 按照预设的压力点数量列表 [2, 3]，为每种情况生成指定数量的样本。
# 2. 对每个样本，随机生成多个受压区域，每个区域具有随机的形状、位置和电导率。
# 3. 对每个生成的电导率分布图(Y)，调用已验证的有限元求解器计算出对应的15维电阻向量(X)。
# 4. 将所有生成的 (X, Y) 数据对，分别保存到 X_data.npy 和 Y_data.npy 文件中。
#
# 使用方法:
# 1. 将此脚本放置在 'dataset/multi_point/' 目录下。
# 2. 确保 'structured_mesh_grid_new/' 文件夹与 'multi_point' 文件夹在同一级目录下。
# 3. 运行此脚本: python create_dataset_multi_point.py
# -----------------------------------------------------------------------------

# 导入必要的库
import numpy as np
import meshio
import os
import random
import tqdm  # 引入 tqdm 来显示进度条

# 显式地从 skfem 的不同模块中导入所有需要的类和函数
from skfem import (MeshHex, ElementHex1, Basis)
from skfem.assembly import asm, BilinearForm
from skfem.utils import solve
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve

# =============================================================================
# --- 1. 超参数定义 ---
# =============================================================================

# a) 数据集相关参数
OUTPUT_DIR = "."  # 将输出目录设为当前文件夹
# 【新】定义我们想要模拟的压力点的数量
NUM_POINTS_TO_SIMULATE = [2, 3]  # 先生成2个压力点的情况，再生成3个压力点的情况
SAMPLES_PER_CASE = 500      # 为每种情况（2点或3点）都生成500个样本

# 【新】定义每个随机压力点的可能形状 (height, width)
POSSIBLE_SHAPES = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)] 

# b) 物理模型和有限元模型参数 (与之前保持一致)
BASE_CONDUCTIVITY = 1.0
CONDUCTIVITY_RATIO_RANGE = (2.0, 20.0)
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

@BilinearForm
def stiffness(u, v, w):
    """
    一个通用的“配方”，它期望通过关键字参数 'sigma' 接收电导率。
    """
    return w.sigma * dot(grad(u), grad(v))

def create_random_multi_point_map(num_points):
    """
    根据给定的压力点数量，生成一个随机的多点压力电导率分布图。
    
    Args:
        num_points (int): 要生成的压力点的数量。
    
    Returns:
        numpy.ndarray: 一个144维的向量，代表每个微元的电导率。
    """
    # a. 初始化一个均匀的背景电导率图
    sigma_bg = BASE_CONDUCTIVITY + np.random.uniform(-0.05, 0.05)
    conductivity_map = np.full(TOTAL_ELEMENTS, sigma_bg)

    # b. 循环 num_points 次，依次添加每个压力点
    for _ in range(num_points):
        
        # i. 为当前这个压力点，随机选择一个形状
        shape = random.choice(POSSIBLE_SHAPES)
        
        # ii. 为当前这个压力点，随机确定一个位置 (x_loc, y_loc)
        shape_height, shape_width = shape
        max_x = NX - shape_width
        max_y = NY - shape_height
        x_loc = np.random.randint(0, max_x + 1)
        y_loc = np.random.randint(0, max_y + 1)
        
        # iii. 为当前这个压力点，随机确定一个电导率幅值
        ratio = np.random.uniform(*CONDUCTIVITY_RATIO_RANGE)
        sigma_pressed = sigma_bg * ratio
        
        # iv. 将这个压力点“画”在电导率图上
        for j in range(y_loc, y_loc + shape_height):
            for i in range(x_loc, x_loc + shape_width):
                k = 0 # 假设是单层
                element_index = k * NY * NX + j * NX + i
                conductivity_map[element_index] = sigma_pressed
                
    return conductivity_map


def solve_fem_for_map(conductivity_map, basis, electrode_nodes):
    """
    【v12 核心逻辑 - 优化版】
    对于一个给定的电导率分布, 计算出其对应的15维电阻测量向量。
    """
    # 1. 初始化一个大小与节点数量(basis.N)相同的数组，用于存储每个节点的电导率。
    #    基础值现在根据 conductivity_map 中的最小值来设定，以适应各种情况。
    min_conductivity = np.min(conductivity_map)
    nodal_conductivity = np.full(basis.N, min_conductivity, dtype=np.float64)

    # 2. 遍历所有单元，将其电导率值“传播”给其构成节点。
    #    如果一个节点被多个不同电导率的单元共享，它会取最大值。
    #    这个方法对于所有情况（阶跃、高斯、多点）都是稳健的。
    for i in range(TOTAL_ELEMENTS):
        nodes_of_element = basis.mesh.t[:, i]
        current_conductivity = conductivity_map[i]
        # 更新这些节点上的电导率，只有在遇到更大的值时才更新
        nodal_conductivity[nodes_of_element] = np.maximum(
            nodal_conductivity[nodes_of_element], 
            current_conductivity
        )
    
    # 3. 在 P1 基上，使用这个新的逐节点数组来创建一个与电势场完全兼容的电导率场。
    sigma_field = basis.interpolate(nodal_conductivity)

    # 4. 在调用 asm 时，将这个兼容的 DiscreteField 对象作为参数传递。
    A = asm(stiffness, basis, sigma=sigma_field)
    
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

    # --- b. 初始化基函数 ---
    element = ElementHex1()
    basis = Basis(mesh, element)
    
    # --- c. 准备数据存储 ---
    X_data_list = []
    Y_data_list = []
    
    # --- d. 主循环，生成数据 ---
    total_samples_to_run = SAMPLES_PER_CASE * len(NUM_POINTS_TO_SIMULATE)
    
    with tqdm.tqdm(total=total_samples_to_run, desc="总进度") as pbar_total:
        for num_points in NUM_POINTS_TO_SIMULATE:
            pbar_total.set_description(f"生成 {num_points} 点压力样本")
            for i in range(SAMPLES_PER_CASE):
                # i. 生成一个随机的多点压力电导率图 (Y)
                conductivity_map = create_random_multi_point_map(num_points)
                
                # ii. 调用求解器，计算对应的电阻向量 (X)
                resistance_vector = solve_fem_for_map(conductivity_map, basis, electrode_nodes)
                
                # iii. 将这对数据 (X, Y) 追加到列表中
                X_data_list.append(resistance_vector)
                Y_data_list.append(conductivity_map)
                
                pbar_total.update(1)

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

