"""
钙钛矿材料空位分析模块
主要功能：
1. 空位结构生成
2. 空位形成能计算
3. 离子传输路径分析
4. 稳定性评估
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import VoronoiNN
from typing import List, Dict, Tuple, Optional
from vasp_interface import VaspInterface

class VacancyAnalyzer:
    """空位分析器类"""
    
    def __init__(self,
                 structure_file: str,
                 vacancy_types: List[str],
                 vasp_work_dir: str = "vasp_calculations",
                 pseudo_dir: str = "/opt/vasp/potentials",
                 vasp_cmd: str = "vasp_std"):
        """
        初始化空位分析器
        
        Args:
            structure_file: 结构CIF文件路径
            vacancy_types: 空位类型列表，如 ["O", "Li", "La"]
            vasp_work_dir: VASP计算工作目录
            pseudo_dir: 赝势文件目录
            vasp_cmd: VASP执行命令
        """
        self.base_structure = Structure.from_file(structure_file)
        self.vacancy_types = vacancy_types
        self.vacancy_structures = {}
        
        # 初始化VASP接口
        self.vasp = VaspInterface(
            work_dir=vasp_work_dir,
            pseudo_dir=pseudo_dir,
            vasp_cmd=vasp_cmd
        )
    
    def analyze_coordination(self) -> Dict:
        """分析原子配位环境"""
        # 使用VoronoiNN分析配位环境
        voronoi = VoronoiNN()
        coordination_info = {}
        
        for site_idx, site in enumerate(self.base_structure):
            # 获取近邻原子
            try:
                neighbors = voronoi.get_nn_info(self.base_structure, site_idx)
                coordination_info[site_idx] = {
                    "element": site.species_string,
                    "coordination_number": len(neighbors),
                    "neighbor_elements": [n["site"].species_string for n in neighbors]
                }
            except Exception as e:
                print(f"Warning: 分析位点 {site_idx} 时出错: {e}")
                coordination_info[site_idx] = {
                    "element": site.species_string,
                    "coordination_number": 0,
                    "neighbor_elements": []
                }
        
        return coordination_info
    
    def find_vacancy_sites(self, element: str) -> List[int]:
        """
        确定可能的空位位置
        
        Args:
            element: 目标元素
            
        Returns:
            适合形成空位的位点索引列表
        """
        # 找到所有目标元素的位点
        target_sites = [i for i, site in enumerate(self.base_structure)
                       if site.species_string == element]
        
        # 分析每个位点的重要性
        site_importance = []
        voronoi = VoronoiNN()
        
        for idx in target_sites:
            try:
                # 获取配位环境
                neighbors = voronoi.get_nn_info(self.base_structure, idx)
                
                # 评估位点重要性（基于配位数和近邻类型）
                importance = len(neighbors)
                if element == "O":
                    # 氧空位优先考虑Li-O-Li路径
                    li_neighbors = sum(1 for n in neighbors 
                                    if "Li" in n["site"].species_string)
                    importance *= li_neighbors
                    
                site_importance.append((idx, importance))
            except Exception as e:
                print(f"Warning: 分析位点 {idx} 时出错: {e}")
                site_importance.append((idx, 0))
        
        # 按重要性排序
        return [idx for idx, _ in sorted(site_importance, 
                                       key=lambda x: x[1],
                                       reverse=True)]
    
    def generate_vacancy_structures(self) -> Dict[str, Structure]:
        """生成空位结构"""
        for element in self.vacancy_types:
            # 获取可能的空位位点
            vacancy_sites = self.find_vacancy_sites(element)
            
            # 对每个位点生成空位结构
            for site_idx in vacancy_sites[:3]:  # 选择前3个最重要的位点
                # 创建空位结构
                defect_structure = self.base_structure.copy()
                defect_structure.remove_sites([site_idx])
                
                # 保存结构
                key = f"{element}_vac_{site_idx}"
                self.vacancy_structures[key] = defect_structure
                
        return self.vacancy_structures
    
    def analyze_stability(self) -> Dict[str, float]:
        """
        分析空位结构的稳定性
        
        Returns:
            各空位结构的稳定性评分
        """
        stability_scores = {}
        for name, structure in self.vacancy_structures.items():
            # 计算空位形成能
            formation_energy = self._calculate_formation_energy(structure)
            
            # 评估结构畸变
            distortion = self._evaluate_structural_distortion(structure)
            
            # 综合评分
            stability_score = self._calculate_stability_score(formation_energy, distortion)
            stability_scores[name] = stability_score
            
        return stability_scores
    
    def evaluate_conductivity(self) -> Dict[str, float]:
        """
        评估空位结构的离子电导率
        
        Returns:
            各空位结构的电导率评估结果
        """
        conductivity_results = {}
        for name, structure in self.vacancy_structures.items():
            # 分析离子迁移路径
            migration_paths = self._analyze_migration_paths(structure)
            
            # 计算迁移势垒
            barriers = self._calculate_migration_barriers(structure, migration_paths)
            
            # 估算电导率
            conductivity = self._estimate_conductivity(barriers)
            conductivity_results[name] = conductivity
            
        return conductivity_results
    
    def _calculate_formation_energy(self, structure: Structure) -> float:
        """
        计算空位形成能
        
        Args:
            structure: 空位结构
            
        Returns:
            形成能 (eV)
        """
        try:
            # 设置化学势
            chemical_potentials = self._get_chemical_potentials()
            
            # 使用VASP接口计算形成能
            formation_energy = self.vasp.calculate_formation_energy(
                defect_structure=structure,
                perfect_structure=self.base_structure,
                chemical_potentials=chemical_potentials
            )
            
            return formation_energy
            
        except Exception as e:
            print(f"Warning: 计算形成能时出错: {e}")
            return float('inf')
    
    def _get_chemical_potentials(self) -> Dict[str, float]:
        """获取元素的化学势"""
        # 这里可以根据实际情况设置化学势
        potentials = {
            "O": -4.0,
            "Li": -1.9,
            "La": -4.9,
            "Ti": -7.5,
            "Nb": -7.0
        }
        return potentials
    
    def _evaluate_structural_distortion(self, structure: Structure) -> float:
        """
        评估结构畸变程度
        
        Args:
            structure: 空位结构
            
        Returns:
            畸变程度 (0-1)
        """
        try:
            # 准备VASP计算
            self.vasp.prepare_input_files(
                structure=structure,
                calc_type="relax",
                ISIF=2,  # 只优化原子位置，保持晶胞固定
                NSW=50   # 最大离子步数
            )
            
            # 运行计算
            if not self.vasp.run_calculation():
                return 1.0
                
            # 获取结果
            results = self.vasp.get_results()
            
            # 分析结构变化
            if not results.get("is_converged", False):
                return 1.0
                
            # 计算原子位移
            forces = np.array(results.get("forces", []))
            max_force = np.max(np.abs(forces))
            
            # 归一化处理 (0表示无畸变，1表示严重畸变)
            distortion = min(1.0, max_force / 5.0)  # 5.0 eV/Å作为参考值
            
            return distortion
            
        except Exception as e:
            print(f"Warning: 评估结构畸变时出错: {e}")
            return 1.0
    
    def _calculate_stability_score(self, 
                                 formation_energy: float, 
                                 distortion: float) -> float:
        """计算综合稳定性评分"""
        # 权重可以根据实际情况调整
        energy_weight = 0.7
        distortion_weight = 0.3
        
        # 归一化处理
        normalized_energy = formation_energy / 5.0  # 假设5.0 eV为参考值
        normalized_distortion = distortion / 0.5  # 假设0.5 Å为参考值
        
        return (energy_weight * normalized_energy + 
                distortion_weight * normalized_distortion)
    
    def _analyze_migration_paths(self, structure: Structure) -> List[Tuple]:
        """
        分析可能的离子迁移路径
        
        Args:
            structure: 空位结构
            
        Returns:
            迁移路径列表，每个元素为(起点索引，终点索引)
        """
        try:
            paths = []
            voronoi = VoronoiNN()
            
            # 找到所有空位（通过比较原始结构和当前结构）
            vacancies = []
            original_sites = set((site.coords.tolist(), site.species_string) 
                               for site in self.base_structure)
            current_sites = set((site.coords.tolist(), site.species_string) 
                              for site in structure)
            
            # 找出缺失的位点
            missing_sites = original_sites - current_sites
            
            # 对每个缺失位点，找到最近的原始结构位点索引
            for missing_coords, missing_element in missing_sites:
                min_dist = float('inf')
                vacancy_idx = -1
                
                for i, site in enumerate(self.base_structure):
                    if site.species_string == missing_element:
                        dist = np.linalg.norm(np.array(site.coords) - np.array(missing_coords))
                        if dist < min_dist:
                            min_dist = dist
                            vacancy_idx = i
                            
                if vacancy_idx >= 0:
                    vacancies.append(vacancy_idx)
            
            # 对每个空位，寻找可能的迁移路径
            for vac_idx in vacancies:
                # 获取近邻原子（使用原始结构分析）
                neighbors = voronoi.get_nn_info(self.base_structure, vac_idx)
                
                # 寻找合适的迁移离子
                for neighbor in neighbors:
                    if neighbor["site"].species_string in ["Li", "O"]:  # 主要考虑Li和O的迁移
                        # 检查该位点在当前结构中是否存在
                        neighbor_coords = neighbor["site"].coords.tolist()
                        if any(np.allclose(site.coords, neighbor_coords) for site in structure):
                            paths.append((vac_idx, neighbor["site_index"]))
            
            return paths
            
        except Exception as e:
            print(f"Warning: 分析迁移路径时出错: {e}")
            return []
    
    def _calculate_migration_barriers(self, 
                                    structure: Structure,
                                    paths: List[Tuple]) -> List[float]:
        """
        计算迁移势垒
        
        Args:
            structure: 空位结构
            paths: 迁移路径列表
            
        Returns:
            迁移势垒列表 (eV)
        """
        try:
            barriers = []
            
            for start_idx, end_idx in paths:
                # 创建初始和终态结构
                initial = structure.copy()
                final = structure.copy()
                
                # 移动离子到终态位置
                # 这里需要根据实际情况设计离子迁移路径
                
                # 计算NEB
                barrier_info = self.vasp.calculate_migration_barrier(
                    initial_structure=initial,
                    final_structure=final,
                    n_images=5
                )
                
                barriers.append(barrier_info["barrier"])
            
            return barriers if barriers else [float('inf')]
            
        except Exception as e:
            print(f"Warning: 计算迁移势垒时出错: {e}")
            return [float('inf')]
    
    def _estimate_conductivity(self, barriers: List[float]) -> float:
        """估算电导率"""
        # 使用Arrhenius方程估算
        T = 300  # 室温
        kb = 8.617333262e-5  # 玻尔兹曼常数 (eV/K)
        
        # 取最低的迁移势垒
        min_barrier = min(barriers) if barriers else float('inf')
        
        # 估算电导率
        conductivity = np.exp(-min_barrier / (kb * T))
        return conductivity 