# -----------------------------------------------------------------------------
# 数据集生成脚本: 随机像素点 (create_dataset_random_pixels.py)
# 
# 【v12 核心集成版】
# 功能:
# 1. 生成指定数量的样本。
# 2. 对每个样本，在12x12的电导率图上，随机选择N个点，并为每个点赋予随机的电导率值。
# 3. 对每个生成的电导率分布图(Y)，调用已验证的有限元求解器计算出对应的15维电阻向量(X)。
# 4. 将所有生成的 (X, Y) 数据对，分别保存到 X_data.npy 和 Y_data.npy 文件中。
#
# 使用方法:
# 1. 将此脚本放置在 'dataset/random_pixels/' 目录下。
# 2. 确保 'structured_mesh_grid_new/' 文件夹与 'random_pixels' 文件夹在同一级目录下。
# 3. 运行此脚本: python create_dataset_random_pixels.py
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
TOTAL_SAMPLES_TO_GENERATE = 1000 # 您期望生成的样本总数

# 【新】定义每个样本中，高电导率“热点”像素的数量范围
NUM_HOT_PIXELS_RANGE = (5, 25) # 每个样本随机包含5到25个热点

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

def create_random_pixel_map():
    """
    生成一个具有随机数量、随机位置、随机幅值的高电导率点的电导率图。
    
    Returns:
        numpy.ndarray: 一个144维的向量，代表每个微元的电导率。
    """
    # a. 初始化一个均匀的背景电导率图
    sigma_bg = BASE_CONDUCTIVITY + np.random.uniform(-0.05, 0.05)
    conductivity_map = np.full(TOTAL_ELEMENTS, sigma_bg)
    
    # b. 随机决定本样本要包含多少个高电导率点
    num_hot_pixels = np.random.randint(NUM_HOT_PIXELS_RANGE[0], NUM_HOT_PIXELS_RANGE[1] + 1)
    
    # c. 从144个位置中，无放回地随机抽取 num_hot_pixels 个位置的索引
    hot_pixel_indices = np.random.choice(TOTAL_ELEMENTS, size=num_hot_pixels, replace=False)
    
    # d. 为每一个被选中的“热点”像素，独立地赋予一个随机的高电导率值
    for index in hot_pixel_indices:
        ratio = np.random.uniform(*CONDUCTIVITY_RATIO_RANGE)
        sigma_pressed = sigma_bg * ratio
        conductivity_map[index] = sigma_pressed
        
    return conductivity_map


def solve_fem_for_map(conductivity_map, basis, electrode_nodes):
    """
    【v12 核心逻辑 - 优化版】
    对于一个给定的电导率分布, 计算出其对应的15维电阻测量向量。
    """
    # 1. 初始化一个大小与节点数量(basis.N)相同的数组，用于存储每个节点的电导率。
    min_conductivity = np.min(conductivity_map)
    nodal_conductivity = np.full(basis.N, min_conductivity, dtype=np.float64)

    # 2. 遍历所有单元，将其电导率值“传播”给其构成节点。
    #    如果一个节点被多个不同电导率的单元共享，它会取最大值。
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
    print(f"开始生成 {TOTAL_SAMPLES_TO_GENERATE} 个随机像素点样本...")
    for i in tqdm.trange(TOTAL_SAMPLES_TO_GENERATE, desc="生成进度"):
        # i. 生成一个随机的电导率图 (Ground Truth Y)
        conductivity_map = create_random_pixel_map()
        
        # ii. 调用求解器，计算对应的电阻向量 (Input X)
        resistance_vector = solve_fem_for_map(conductivity_map, basis, electrode_nodes)
        
        # iii. 将这对数据 (X, Y) 追加到列表中
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

