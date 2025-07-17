#!/usr/bin/env python3
"""
结构与性能可视化模块
提供3D结构可视化和性能数据可视化功能
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StructureVisualizer:
    """结构与性能可视化类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.style_config = {
            'background_color': 'white',
            'atom_style': 'sphere',
            'bond_style': 'stick'
        }
        
        # 设置matplotlib样式
        plt.style.use('seaborn')
        
    def _ensure_output_dir(self, output_file: str) -> None:
        """确保输出目录存在"""
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
    
    def visualize_structure_3d(self, structure_file: str, 
                             output_html: Optional[str] = None,
                             width: int = 800, 
                             height: int = 600) -> Optional[py3Dmol.view]:
        """
        使用py3Dmol可视化结构的3D表示
        
        Args:
            structure_file: CIF文件路径
            output_html: 输出HTML文件路径（可选）
            width: 视图宽度
            height: 视图高度
            
        Returns:
            py3Dmol.view对象或None（如果发生错误）
        """
        try:
            if not os.path.exists(structure_file):
                raise FileNotFoundError(f"结构文件不存在: {structure_file}")
            
            # 解析CIF文件
            structure = Structure.from_file(structure_file)
            
            # 创建3D视图
            view = py3Dmol.view(width=width, height=height)
            
            # 添加结构到视图
            xyz_str = structure.to(fmt="xyz")
            view.addModel(xyz_str, "xyz")
            view.setStyle({'sphere':{'colorscheme':'Jmol','scale':0.3},
                          'stick':{'radius':0.2}})
            
            # 设置视图属性
            view.zoomTo()
            view.setBackgroundColor(self.style_config['background_color'])
            
            # 保存为HTML文件
            if output_html:
                self._ensure_output_dir(output_html)
                view.save(output_html)
                logger.info(f"已保存3D结构可视化结果到: {output_html}")
                
            return view
            
        except Exception as e:
            logger.error(f"3D结构可视化失败: {str(e)}")
            return None
    
    def plot_property_distribution(self, 
                                 data: Dict[str, float],
                                 property_name: str,
                                 output_file: Optional[str] = None,
                                 plot_type: str = 'bar') -> None:
        """
        绘制性能数据分布图
        
        Args:
            data: 性能数据字典 {材料名: 性能值}
            property_name: 性能属性名称
            output_file: 输出文件路径（可选）
            plot_type: 图表类型 ('bar', 'line', 'scatter')
        """
        try:
            if not data:
                raise ValueError("输入数据为空")
                
            plt.figure(figsize=(12, 6))
            
            if plot_type == 'bar':
                sns.barplot(x=list(data.keys()), y=list(data.values()))
            elif plot_type == 'line':
                plt.plot(list(data.keys()), list(data.values()), marker='o')
            elif plot_type == 'scatter':
                plt.scatter(range(len(data)), list(data.values()))
            else:
                raise ValueError(f"不支持的图表类型: {plot_type}")
                
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Materials')
            plt.ylabel(property_name)
            plt.title(f'{property_name} Distribution')
            
            if output_file:
                self._ensure_output_dir(output_file)
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                logger.info(f"已保存分布图到: {output_file}")
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制性能分布图失败: {str(e)}")
            plt.close()
        
    def create_heatmap(self,
                      data_matrix: np.ndarray,
                      x_labels: List[str],
                      y_labels: List[str],
                      title: str,
                      output_file: Optional[str] = None) -> None:
        """
        创建热图
        
        Args:
            data_matrix: 数据矩阵
            x_labels: X轴标签
            y_labels: Y轴标签
            title: 图表标题
            output_file: 输出文件路径（可选）
        """
        try:
            if data_matrix.size == 0:
                raise ValueError("输入数据矩阵为空")
            if len(x_labels) != data_matrix.shape[1] or len(y_labels) != data_matrix.shape[0]:
                raise ValueError("标签数量与数据矩阵维度不匹配")
                
            plt.figure(figsize=(10, 8))
            sns.heatmap(data_matrix, 
                       xticklabels=x_labels,
                       yticklabels=y_labels,
                       annot=True,
                       cmap='viridis',
                       fmt='.2f')
            
            plt.title(title)
            if output_file:
                self._ensure_output_dir(output_file)
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                logger.info(f"已保存热图到: {output_file}")
            plt.close()
            
        except Exception as e:
            logger.error(f"创建热图失败: {str(e)}")
            plt.close()
        
    def plot_dynamic_performance(self,
                               time_series_data: Dict[str, List[float]],
                               property_name: str,
                               output_file: Optional[str] = None) -> None:
        """
        创建动态性能图表
        
        Args:
            time_series_data: 时间序列数据 {材料名: [性能值列表]}
            property_name: 性能属性名称
            output_file: 输出文件路径（可选）
        """
        try:
            if not time_series_data:
                raise ValueError("输入数据为空")
                
            # 检查所有序列长度是否一致
            lengths = [len(values) for values in time_series_data.values()]
            if len(set(lengths)) > 1:
                raise ValueError("所有时间序列必须具有相同的长度")
                
            fig = go.Figure()
            
            for material, values in time_series_data.items():
                fig.add_trace(go.Scatter(
                    y=values,
                    name=material,
                    mode='lines+markers'
                ))
                
            fig.update_layout(
                title=f'Dynamic {property_name} Performance',
                xaxis_title='Cycle Number',
                yaxis_title=property_name,
                hovermode='x unified'
            )
            
            if output_file:
                self._ensure_output_dir(output_file)
                fig.write_html(output_file)
                logger.info(f"已保存动态性能图表到: {output_file}")
                
        except Exception as e:
            logger.error(f"创建动态性能图表失败: {str(e)}")
            
    def plot_multiple_properties(self,
                               data: Dict[str, Dict[str, float]],
                               properties: List[str],
                               output_file: Optional[str] = None) -> None:
        """
        创建多属性对比图
        
        Args:
            data: 多属性数据 {材料名: {属性名: 值}}
            properties: 要对比的属性列表
            output_file: 输出文件路径（可选）
        """
        try:
            if not data or not properties:
                raise ValueError("输入数据或属性列表为空")
                
            # 检查所有材料是否都具有所有指定的属性
            for material, props in data.items():
                missing_props = set(properties) - set(props.keys())
                if missing_props:
                    raise ValueError(f"材料 {material} 缺少以下属性: {missing_props}")
                    
            fig = make_subplots(rows=len(properties), cols=1,
                               subplot_titles=properties)
            
            for i, prop in enumerate(properties, 1):
                prop_data = {mat: values[prop] for mat, values in data.items()}
                
                fig.add_trace(
                    go.Bar(x=list(prop_data.keys()),
                          y=list(prop_data.values()),
                          name=prop),
                    row=i, col=1
                )
                
            fig.update_layout(height=300*len(properties),
                            showlegend=False,
                            title_text="Multiple Properties Comparison")
            
            if output_file:
                self._ensure_output_dir(output_file)
                fig.write_html(output_file)
                logger.info(f"已保存多属性对比图到: {output_file}")
                
        except Exception as e:
            logger.error(f"创建多属性对比图失败: {str(e)}")

def main():
    """主函数用于测试"""
    try:
        visualizer = StructureVisualizer()
        
        # 创建输出目录
        os.makedirs("visualization_results", exist_ok=True)
        
        # 测试3D结构可视化
        cif_file = "data/01Li-La-Ti–O₃ 主族， NbZrAlGa 衍生物/LaTiO3.cif"
        if os.path.exists(cif_file):
            view = visualizer.visualize_structure_3d(
                cif_file, 
                "visualization_results/structure_3d.html"
            )
            
        # 测试性能数据可视化
        test_data = {
            "Material1": 0.85,
            "Material2": 0.92,
            "Material3": 0.78,
            "Material4": 0.95
        }
        visualizer.plot_property_distribution(
            test_data, 
            "Conductivity (S/cm)", 
            "visualization_results/conductivity_distribution.png"
        )
        
        # 测试热图
        test_matrix = np.random.rand(4, 4)
        materials = ["Mat1", "Mat2", "Mat3", "Mat4"]
        properties = ["Prop1", "Prop2", "Prop3", "Prop4"]
        visualizer.create_heatmap(
            test_matrix, 
            materials, 
            properties,
            "Property Correlation Matrix",
            "visualization_results/correlation_heatmap.png"
        )
        
        logger.info("所有测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main() 