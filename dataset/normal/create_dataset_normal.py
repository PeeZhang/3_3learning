# -----------------------------------------------------------------------------
# 数据集生成脚本: 正常工况 (create_dataset_normal.py)
#
# 功能:
# 1. 生成指定数量的、模拟法兰垫片在正常压紧状态下的样本。
# 2. 在这些样本中，整个材料的电导率在一个较高的范围内均匀波动。
# 3. 对每个生成的电导率分布图(Y)，调用已验证的有限元求解器计算出对应的电阻向量(X)。
# -----------------------------------------------------------------------------

import numpy as np
import meshio
import os
import tqdm

from skfem import (MeshHex, ElementHex1, Basis)
from skfem.assembly import asm, BilinearForm
from skfem.utils import solve
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve

# =============================================================================
# --- 1. 超参数定义 ---
# =============================================================================

# a) 数据集相关参数
OUTPUT_DIR = "."
TOTAL_SAMPLES_TO_GENERATE = 2000

# b) 物理模型相关参数
# 【新】定义正常工况下的高电导率范围
NORMAL_CONDUCTIVITY_RATIO_RANGE = (15.0, 25.0) 
BASE_CONDUCTIVITY = 1.0 # 仅作为参考，实际值由上述范围决定

# c) 有限元模型相关参数
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
    return w.sigma * dot(grad(u), grad(v))

def create_normal_conductivity_map():
    """
    生成一个模拟正常均匀高压状态的电导率分布图。
    """
    # 随机选择一个在该范围内的高斯基础值
    sigma_base_ratio = np.random.uniform(*NORMAL_CONDUCTIVITY_RATIO_RANGE)
    sigma_base = BASE_CONDUCTIVITY * sigma_base_ratio
    
    # 在这个高斯基础上增加微小的随机扰动
    conductivity_map = np.full(TOTAL_ELEMENTS, sigma_base)
    noise = np.random.uniform(-0.5, 0.5, size=TOTAL_ELEMENTS)
    conductivity_map += noise
    
    return conductivity_map

def solve_fem_for_map(conductivity_map, basis, electrode_nodes):
    """
    【v12 核心逻辑】
    对于一个给定的电导率分布, 计算出其对应的15维电阻测量向量。
    """
    nodal_conductivity = np.full(basis.N, BASE_CONDUCTIVITY, dtype=np.float64)
    high_conductivity_elements_indices = np.where(conductivity_map >= BASE_CONDUCTIVITY)[0]

    if high_conductivity_elements_indices.size > 0:
        # 优化：直接将每个单元的值映射到其节点，取最大值
        for i in range(TOTAL_ELEMENTS):
            nodes_of_element = basis.mesh.t[:, i]
            current_conductivity = conductivity_map[i]
            nodal_conductivity[nodes_of_element] = np.maximum(
                nodal_conductivity[nodes_of_element], current_conductivity
            )
            
    sigma_field = basis.interpolate(nodal_conductivity)
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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    mesh_path = os.path.join(MESH_DIR, MESH_FILENAME)
    nodes_path = os.path.join(MESH_DIR, NODES_FILENAME)
    try:
        m = meshio.read(mesh_path)
        mesh = MeshHex(m.points.T, m.cells_dict['hexahedron'].T)
        electrode_nodes = np.load(nodes_path)
    except Exception as e:
        print(f"错误：无法加载所需文件。请确保路径 '{MESH_DIR}' 正确。错误信息: {e}")
        exit()

    element = ElementHex1()
    basis = Basis(mesh, element)
    
    X_data_list, Y_data_list = [], []
    
    print(f"开始生成 {TOTAL_SAMPLES_TO_GENERATE} 个“正常工况”样本...")
    for _ in tqdm.trange(TOTAL_SAMPLES_TO_GENERATE, desc="生成进度"):
        conductivity_map = create_normal_conductivity_map()
        resistance_vector = solve_fem_for_map(conductivity_map, basis, electrode_nodes)
        
        X_data_list.append(resistance_vector)
        Y_data_list.append(conductivity_map)

    print("\n所有数据生成完毕，正在保存...")
    
    X_data = np.array(X_data_list)
    Y_data = np.array(Y_data_list)
    
    x_path = os.path.join(OUTPUT_DIR, "X_data.npy")
    y_path = os.path.join(OUTPUT_DIR, "Y_data.npy")
    np.save(x_path, X_data)
    np.save(y_path, Y_data)
    
    print(f"数据集保存成功！")
    print(f"  - 输入数据 (X): {X_data.shape} -> 已保存至 {x_path}")
    print(f"  - 标签数据 (Y): {Y_data.shape} -> 已保存至 {y_path}")
