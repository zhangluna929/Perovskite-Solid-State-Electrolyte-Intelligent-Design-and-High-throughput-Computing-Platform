"""
迁移势垒计算演示脚本
展示如何计算和分析钙钛矿材料中的离子迁移势垒
"""

import os
import logging
import json
from pymatgen.core import Structure
from migration_barrier_analysis import MigrationBarrierAnalyzer
from vasp_interface import VaspInterface
from structure_visualizer import StructureVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 设置工作目录
    work_dir = "migration_barrier_demo"
    os.makedirs(work_dir, exist_ok=True)
    
    # 初始化VASP接口
    vasp_interface = VaspInterface(
        work_dir=os.path.join(work_dir, "vasp"),
        pseudo_dir="/path/to/your/pseudopotentials",  # 请修改为实际的赝势文件路径
        vasp_cmd="mpirun -np 4 vasp_std"  # 根据实际环境修改
    )
    
    # 初始化迁移势垒分析器
    analyzer = MigrationBarrierAnalyzer(vasp_interface)
    
    # 加载示例结构（以LiNbO3为例）
    structure = Structure.from_file("data/02经典钙钛矿锂氧族 (TaNb 系)/LiNbO3.cif")
    
    # 1. 分析可能的迁移路径
    logger.info("分析可能的迁移路径...")
    migration_paths = analyzer.find_migration_paths(
        structure,
        migrating_specie="Li",
        max_distance=5.0  # Å
    )
    
    # 2. 计算每个路径的迁移势垒
    results = []
    for i, path in enumerate(migration_paths):
        logger.info(f"计算第{i+1}条路径的迁移势垒...")
        
        # 计算迁移势垒
        barrier_result = analyzer.calculate_migration_barrier(
            structure=structure,
            migration_path=path,
            n_images=5,  # NEB图像数量
            is_climbing_image=True  # 使用CI-NEB方法
        )
        
        # 保存结果
        results.append({
            "path_id": i,
            "start_point": path.start_point.tolist(),
            "end_point": path.end_point.tolist(),
            "barrier_height": barrier_result["barrier"],
            "energy_profile": barrier_result["energy_profile"],
            "is_converged": barrier_result["is_converged"]
        })
        
        # 可视化迁移路径
        if barrier_result["is_converged"]:
            visualizer = StructureVisualizer()
            visualizer.plot_migration_path(
                structure,
                path,
                barrier_result["structures"],
                save_file=os.path.join(work_dir, f"migration_path_{i}.png")
            )
    
    # 3. 保存分析结果
    with open(os.path.join(work_dir, "migration_barriers.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # 4. 分析掺杂对迁移势垒的影响
    logger.info("分析掺杂对迁移势垒的影响...")
    dopant_effects = analyzer.analyze_dopant_effects(
        structure,
        migration_paths[0],  # 选择能量最低的路径
        dopants=["Na", "K", "Rb"],  # 可能的掺杂元素
        dopant_concentrations=[0.05, 0.1, 0.15]  # 掺杂浓度
    )
    
    # 保存掺杂分析结果
    with open(os.path.join(work_dir, "dopant_effects.json"), "w") as f:
        json.dump(dopant_effects, f, indent=2)
    
    # 5. 总结分析结果
    logger.info("生成分析报告...")
    analyzer.generate_report(
        migration_results=results,
        dopant_effects=dopant_effects,
        output_file=os.path.join(work_dir, "migration_analysis_report.md")
    )

if __name__ == "__main__":
    main() 