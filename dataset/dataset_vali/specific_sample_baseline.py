# -----------------------------------------------------------------------------
# 全背景电导率验证脚本 (validate_baseline_sample.py)
#
# 功能:
# 1. 创建一个均匀的、电导率全为背景值的分布图。
# 2. 计算出其对应的15维电阻向量，作为基准参考。
# -----------------------------------------------------------------------------

# 导入必要的库
import numpy as np
import meshio
import os
import matplotlib.pyplot as plt

# 显式地从 skfem 的不同模块中导入所有需要的类和函数
from skfem import (MeshHex, ElementHex1, Basis)
from skfem.assembly import asm, BilinearForm
from skfem.utils import solve
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve

# =============================================================================
# --- 1. 参数和文件路径定义 ---
# =============================================================================

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

# 网格几何参数
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
    创建均匀的基准电导率分布图。
    """
    conductivity_map = np.full(TOTAL_ELEMENTS, BASE_CONDUCTIVITY, dtype=np.float64)
    print("已生成均匀的基准电导率图。")
    return conductivity_map


def solve_fem_for_map(conductivity_map, basis, electrode_nodes):
    """
    对于一个给定的电导率分布，计算出其对应的15维电阻测量向量。
    """
    # 【v12 核心逻辑】手动将 P0（逐单元）数据转换为 P1（逐节点）数据
    nodal_conductivity = np.full(basis.N, BASE_CONDUCTIVITY, dtype=np.float64)
    high_conductivity_elements_indices = np.where(conductivity_map > BASE_CONDUCTIVITY)[0]
    if high_conductivity_elements_indices.size > 0:
        nodes_of_high_conductivity_elements = basis.mesh.t[:, high_conductivity_elements_indices]
        unique_node_indices = np.unique(nodes_of_high_conductivity_elements)
        # 注意：这里我们假设高电导率只有一个值
        pressed_value = np.max(conductivity_map) 
        nodal_conductivity[unique_node_indices] = pressed_value
    
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
    
    # --- c. 生成特定的电导率图并进行计算 ---
    conductivity_map = create_baseline_conductivity_map()
    resistance_vector = solve_fem_for_map(conductivity_map, basis, electrode_nodes)
    
    # --- d. 在控制台输出结果 ---
    print("\n" + "="*30)
    print("--- 仿真结果 (全背景电导率) ---")
    print("="*30)
    print("\n- 输出 X (15维电阻向量):")
    print(f"  {np.array2string(resistance_vector, formatter={'float_kind':lambda x: '%.4f' % x})}")
    print("\n" + "="*30)

