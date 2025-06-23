# -----------------------------------------------------------------------------
# 特定样本验证脚本 (t1_b1交叉区受压) (validate_t1b1_sample.py)
#
# 【最终修正版 v12】
# 通过手动将 P0（逐单元）电导率图转换为 P1（逐节点）数据，
# 然后在 P1 基上创建兼容的电导率场，从根本上解决场不兼容问题。
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

# 使用与COMSOL仿真一致的电导率值
BASE_CONDUCTIVITY = 1.0
PRESSED_CONDUCTIVITY = 20.0

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


def create_t1b1_conductivity_map():
    """
    创建特定的电导率分布图：top_1 和 bottom_1 交叉区域为高电导率。
    """
    conductivity_map = np.full(TOTAL_ELEMENTS, BASE_CONDUCTIVITY, dtype=np.float64)
    # 根据 electrode_nodes.npz 的定义，t1_b1 交叉区域的单元索引为 13, 14, 25, 26
    indices_to_change = [13, 14, 25, 26]
    conductivity_map[indices_to_change] = PRESSED_CONDUCTIVITY
    print("已生成特定电导率图，受压单元索引为:", indices_to_change)
    return conductivity_map


def solve_fem_for_map(conductivity_map, basis, electrode_nodes):
    """
    对于一个给定的电导率分布，计算出其对应的15维电阻测量向量。
    """
    # -----------------------------------------------------------------------------
    # 【最终修正方案 v12】
    # 1. 初始化一个大小与节点数量(basis.N)相同的数组，用于存储每个节点的电导率。
    nodal_conductivity = np.full(basis.N, BASE_CONDUCTIVITY, dtype=np.float64)

    # 2. 找到需要改变电导率的单元的索引。
    high_conductivity_elements_indices = np.where(conductivity_map > BASE_CONDUCTIVITY)[0]

    # 3. 将这些单元的较高电导率值，赋给构成这些单元的所有节点。
    #    basis.mesh.t 的形状是 (nodes_per_element, num_elements)，存储了每个单元的节点索引。
    if high_conductivity_elements_indices.size > 0:
        nodes_of_high_conductivity_elements = basis.mesh.t[:, high_conductivity_elements_indices]
        # 获取所有需要改变的节点的唯一索引，以避免重复操作。
        unique_node_indices = np.unique(nodes_of_high_conductivity_elements)
        # 将这些节点的电导率设置为较高值。
        nodal_conductivity[unique_node_indices] = PRESSED_CONDUCTIVITY
    
    # 4. 在 P1 基上，使用这个新的逐节点数组来创建一个与电势场完全兼容的电导率场。
    sigma_field = basis.interpolate(nodal_conductivity)

    # 5. 在调用 asm 时，将这个兼容的 DiscreteField 对象作为参数传递。
    A = asm(stiffness, basis, sigma=sigma_field)
    # -----------------------------------------------------------------------------

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
# --- 3. 主执行流程 (与之前相同) ---
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
    conductivity_map = create_t1b1_conductivity_map()
    resistance_vector = solve_fem_for_map(conductivity_map, basis, electrode_nodes)
    
    # --- d. 在控制台输出结果 ---
    print("\n" + "="*30)
    print("--- 仿真结果 (t1_b1交叉区受压) ---")
    print("="*30)
    print("\n- 输入 Y (144维电导率向量):")
    print(f"  {np.array2string(conductivity_map, formatter={'float_kind':lambda x: '%.1f' % x}, max_line_width=120)}")
    print("\n- 输出 X (15维电阻向量):")
    print(f"  {np.array2string(resistance_vector, formatter={'float_kind':lambda x: '%.4f' % x})}")
    print("\n" + "="*30)
    
    # --- e. 可视化结果 ---
    try:
        # 尝试设置中文字体，如果失败则忽略
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        print("\n警告：未找到 'SimHei' 字体，图表标题可能显示为方框。")
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('特定样本验证 (top_1 vs bottom_1 交叉区域受压)', fontsize=16)
    conductivity_map_image = conductivity_map.reshape(NY, NX)
    ax1.set_title('电导率分布图 (真值 Y)')
    im = ax1.imshow(conductivity_map_image, cmap='viridis', origin='lower', interpolation='nearest')
    ax1.set_xlabel('X 单元索引')
    ax1.set_ylabel('Y 单元索引')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax2.set_title('15维电阻测量向量 (输入 X)')
    measurement_indices = np.arange(1, len(resistance_vector) + 1)
    ax2.bar(measurement_indices, resistance_vector, color='skyblue', width=0.6)
    ax2.set_xlabel('测量电极对索引 (1-15)')
    ax2.set_ylabel('电阻值 (Ω)')
    ax2.set_xticks(measurement_indices)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("validation_t1b1_sample.png")
    print("\n结果可视化图像已保存至 'validation_t1b1_sample.png'")
    plt.show()

