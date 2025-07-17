#!/usr/bin/env python3
"""
可视化功能演示脚本
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
from structure_visualizer import StructureVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visualization_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def load_analysis_results(results_file: str) -> dict:
    """加载分析结果"""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"结果文件不存在: {results_file}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"结果文件格式错误: {results_file}")
        return {}
    except Exception as e:
        logger.error(f"加载结果文件时发生错误: {str(e)}")
        return {}

def ensure_data_files_exist(file_paths: list) -> bool:
    """检查数据文件是否存在"""
    missing_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"以下文件不存在: {', '.join(missing_files)}")
        return False
    return True

def demo_structure_visualization(visualizer: StructureVisualizer) -> None:
    """演示结构可视化"""
    logger.info("正在生成3D结构可视化...")
    
    # 示例：可视化原始结构和掺杂后的结构
    structures = [
        "data/01Li-La-Ti–O₃ 主族， NbZrAlGa 衍生物/LaTiO3.cif",
        "data/01Li-La-Ti–O₃ 主族， NbZrAlGa 衍生物/Li2La2Ti3O10.cif"
    ]
    
    if not ensure_data_files_exist(structures):
        return
    
    try:
        for i, struct_file in enumerate(structures):
            output_file = f"visualization_results/structure_3d_{i}.html"
            os.makedirs("visualization_results", exist_ok=True)
            view = visualizer.visualize_structure_3d(struct_file, output_file)
            if view is not None:
                logger.info(f"已生成3D结构可视化: {output_file}")
            else:
                logger.error(f"生成3D结构可视化失败: {struct_file}")
    except Exception as e:
        logger.error(f"结构可视化过程中发生错误: {str(e)}")

def demo_property_visualization(visualizer: StructureVisualizer) -> None:
    """演示性能数据可视化"""
    logger.info("正在生成性能数据可视化...")
    
    try:
        # 示例：电导率数据
        conductivity_data = {
            "LaTiO3": 0.85,
            "Li2La2Ti3O10": 0.92,
            "LiLaTi2O6": 0.78,
            "Li3La5Ti6Nb2O26": 0.95
        }
        
        # 生成柱状图
        visualizer.plot_property_distribution(
            conductivity_data,
            "Ionic Conductivity (S/cm)",
            "visualization_results/conductivity_distribution.png",
            "bar"
        )
        
        # 示例：多个性能指标的相关性
        performance_matrix = np.array([
            [1.0, 0.8, 0.6, 0.4],
            [0.8, 1.0, 0.7, 0.5],
            [0.6, 0.7, 1.0, 0.8],
            [0.4, 0.5, 0.8, 1.0]
        ])
        
        properties = ["Conductivity", "Stability", "Formation Energy", "Band Gap"]
        visualizer.create_heatmap(
            performance_matrix,
            properties,
            properties,
            "Performance Correlation Matrix",
            "visualization_results/correlation_heatmap.png"
        )
        
        # 示例：循环性能数据
        cycle_data = {
            "LaTiO3": [0.95, 0.93, 0.91, 0.90, 0.89],
            "Li2La2Ti3O10": [0.98, 0.97, 0.96, 0.95, 0.94],
            "LiLaTi2O6": [0.92, 0.91, 0.90, 0.89, 0.88]
        }
        
        visualizer.plot_dynamic_performance(
            cycle_data,
            "Capacity Retention",
            "visualization_results/cycle_performance.html"
        )
        
        # 示例：多属性对比
        multi_property_data = {
            "LaTiO3": {
                "Conductivity": 0.85,
                "Stability": 0.90,
                "Formation Energy": -2.5
            },
            "Li2La2Ti3O10": {
                "Conductivity": 0.92,
                "Stability": 0.95,
                "Formation Energy": -2.8
            },
            "LiLaTi2O6": {
                "Conductivity": 0.78,
                "Stability": 0.88,
                "Formation Energy": -2.3
            }
        }
        
        properties_to_compare = ["Conductivity", "Stability", "Formation Energy"]
        visualizer.plot_multiple_properties(
            multi_property_data,
            properties_to_compare,
            "visualization_results/multi_property_comparison.html"
        )
        
    except Exception as e:
        logger.error(f"性能数据可视化过程中发生错误: {str(e)}")

def main():
    """主函数"""
    try:
        logger.info("开始生成可视化结果...")
        
        # 创建可视化器实例
        visualizer = StructureVisualizer()
        
        # 创建结果目录
        results_dir = Path("visualization_results")
        results_dir.mkdir(exist_ok=True)
        
        # 演示各种可视化功能
        demo_structure_visualization(visualizer)
        demo_property_visualization(visualizer)
        
        logger.info("所有可视化结果已生成完成！")
        logger.info(f"请查看 {results_dir} 目录下的输出文件。")
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 