# -----------------------------------------------------------------------------
# 数据集生成脚本: 大面积随机形状 (create_dataset_large_area.py)
# 
# 功能:
# 1. 生成指定数量的样本。
# 2. 对每个样本，通过“斑块生长”算法生成一个大面积、连续、形状不规则的高电导率区域。
# 3. 对每个生成的电导率分布图(Y)，调用有限元求解器计算出对应的15维电阻向量(X)。
# 4. 将所有生成的 (X, Y) 数据对，分别保存到 X_data.npy 和 Y_data.npy 文件中。
#
# 使用方法:
# 1. 将此脚本放置在 'dataset/large_area/' 目录下。
# 2. 确保 'structured_mesh_grid_new/' 文件夹与 'large_area' 文件夹在同一级目录下。
# 3. 运行此脚本: python create_dataset_large_area.py
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

# 【新】定义大面积“斑块”的面积（所占微元数量）范围
AREA_SIZE_RANGE = (40, 100) # 每个样本的受压面积在40到100个微元之间随机

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

def get_neighbors(index, width, height):
    """根据一维索引，计算其在二维网格中的邻居（上下左右）的一维索引"""
    neighbors = []
    x, y = index % width, index // width
    # 上
    if y > 0: neighbors.append(index - width)
    # 下
    if y < height - 1: neighbors.append(index + width)
    # 左
    if x > 0: neighbors.append(index - 1)
    # 右
    if x < width - 1: neighbors.append(index + 1)
    return neighbors

def create_random_large_area_map():
    """
    通过“斑块生长”算法，生成一个具有大面积、随机形状高电导率区域的电导率图。
    
    Returns:
        numpy.ndarray: 一个144维的向量，代表每个微元的电导率。
    """
    # a. 初始化背景
    sigma_bg = BASE_CONDUCTIVITY + np.random.uniform(-0.05, 0.05)
    conductivity_map = np.full(TOTAL_ELEMENTS, sigma_bg)
    
    # b. 随机决定目标面积和电导率
    target_area = np.random.randint(AREA_SIZE_RANGE[0], AREA_SIZE_RANGE[1] + 1)
    ratio = np.random.uniform(*CONDUCTIVITY_RATIO_RANGE)
    sigma_pressed = sigma_bg * ratio
    
    # c. 执行“斑块生长”算法
    #   i. 随机选择一个起始“种子”点
    seed_index = np.random.randint(0, TOTAL_ELEMENTS)
    blob = {seed_index}
    conductivity_map[seed_index] = sigma_pressed
    
    #   ii. 将所有可以扩张的邻居放入一个“边界”列表
    frontier = get_neighbors(seed_index, NX, NY)
    
    #   iii. 循环生长，直到达到目标面积
    while len(blob) < target_area and frontier:
        # 从边界中随机选择一个点进行扩张
        chosen_neighbor = random.choice(frontier)
        frontier.remove(chosen_neighbor)
        
        # 如果这个点已经是斑块的一部分，就跳过
        if chosen_neighbor in blob:
            continue
            
        # 将新点加入斑块，并更新电导率图
        blob.add(chosen_neighbor)
        conductivity_map[chosen_neighbor] = sigma_pressed
        
        # 将这个新点的邻居加入到边界列表中，以备下一步扩张
        new_neighbors = get_neighbors(chosen_neighbor, NX, NY)
        for neighbor in new_neighbors:
            if neighbor not in blob:
                frontier.append(neighbor)
                
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

    # --- b. 初始化基函数 ---
    element = ElementHex1()
    basis = Basis(mesh, element)
    
    # --- c. 准备数据存储 ---
    X_data_list = []
    Y_data_list = []
    
    # --- d. 主循环，生成数据 ---
    print(f"开始生成 {TOTAL_SAMPLES_TO_GENERATE} 个大面积随机形状样本...")
    for i in tqdm.trange(TOTAL_SAMPLES_TO_GENERATE, desc="生成进度"):
        # i. 生成一个随机的电导率图 (Y)
        conductivity_map = create_random_large_area_map()
        
        # ii. 定义与此图绑定的刚度矩阵“配方”
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
