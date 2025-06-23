# -----------------------------------------------------------------------------
# 数据集生成脚本: 空白初始数据 (create_dataset_baseline.py)
# 
# 【v12 核心集成版】
# 功能:
# 1. 生成指定数量的“无压力”状态下的样本。
# 2. 在这些样本中，整个材料的电导率是均匀的（带有微小的随机噪声）。
# 3. 对每个生成的电导率分布图(Y)，调用已验证的有限元求解器计算出对应的15维电阻向量(X)。
# 4. 将所有生成的 (X, Y) 数据对，分别保存到 X_data.npy 和 Y_data.npy 文件中。
#
# 使用方法:
# 1. 创建一个新的文件夹 'dataset/baseline/'。
# 2. 将此脚本放置在 'dataset/baseline/' 目录下。
# 3. 确保 'structured_mesh_grid_new/' 文件夹与 'baseline' 文件夹在同一级目录下。
# 4. 运行此脚本: python create_dataset_baseline.py
# -----------------------------------------------------------------------------

# 导入必要的库
import numpy as np
import meshio
import os
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
TOTAL_SAMPLES_TO_GENERATE = 500 # 生成100条空白数据

# b) 物理模型和有限元模型参数
BASE_CONDUCTIVITY = 1.0  # 未受压区域的基准电导率
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

# d) 网格几何参数
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

def create_baseline_conductivity_map():
    """
    生成一个均匀的、无任何异常区域的电导率分布图。
    
    Returns:
        numpy.ndarray: 一个144维的向量，代表每个微元的电导率。
    """
    # 初始化电导率图，只包含带有微小随机扰动的背景值
    sigma_bg = BASE_CONDUCTIVITY + np.random.uniform(-0.05, 0.05)
    conductivity_map = np.full(TOTAL_ELEMENTS, sigma_bg)
    return conductivity_map


def solve_fem_for_map(conductivity_map, basis, electrode_nodes):
    """
    【v12 核心逻辑】
    对于一个给定的电导率分布, 计算出其对应的15维电阻测量向量。
    """
    # 1. 初始化一个大小与节点数量(basis.N)相同的数组，用于存储每个节点的电导率。
    nodal_conductivity = np.full(basis.N, BASE_CONDUCTIVITY, dtype=np.float64)

    # 2. 找到需要改变电导率的单元的索引。
    high_conductivity_elements_indices = np.where(conductivity_map > BASE_CONDUCTIVITY)[0]

    # 3. 将这些单元的较高电导率值，赋给构成这些单元的所有节点。
    if high_conductivity_elements_indices.size > 0:
        nodes_of_high_conductivity_elements = basis.mesh.t[:, high_conductivity_elements_indices]
        unique_node_indices = np.unique(nodes_of_high_conductivity_elements)
        # 从原始图中获取实际的、可能带有噪声的高电导率值
        pressed_value = np.max(conductivity_map)
        nodal_conductivity[unique_node_indices] = pressed_value
    
    # 4. 在 P1 基上，使用这个新的逐节点数组来创建一个与电势场完全兼容的电导率场。
    sigma_field = basis.interpolate(nodal_conductivity)

    # 5. 在调用 asm 时，将这个兼容的 DiscreteField 对象作为参数传递。
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
    print(f"开始生成 {TOTAL_SAMPLES_TO_GENERATE} 个空白（无压力）样本...")
    for i in tqdm.trange(TOTAL_SAMPLES_TO_GENERATE, desc="生成进度"):
        # i. 生成一个均匀的电导率图 (Y)
        conductivity_map = create_baseline_conductivity_map()
        
        # ii. 调用求解器，计算对应的电阻向量 (X)
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

