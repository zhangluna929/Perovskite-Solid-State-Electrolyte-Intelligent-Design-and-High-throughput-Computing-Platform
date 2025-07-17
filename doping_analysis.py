"""
钙钛矿材料掺杂分析模块
主要功能：
1. 掺杂结构生成
2. 掺杂形成能计算
3. 电子结构分析
4. 性能评估
"""

import numpy as np
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import VoronoiNN
from typing import List, Dict, Tuple, Optional
from vasp_interface import VaspInterface
from ml_doping_predictor import MLDopingSitePredictor
import os

class DopingAnalyzer:
    """掺杂分析器类"""
    
    def __init__(self, 
                 structure_file: str,
                 doping_elements: List[str],
                 concentration_range: List[float],
                 vasp_work_dir: str = "vasp_calculations",
                 pseudo_dir: str = "/opt/vasp/potentials",
                 vasp_cmd: str = "vasp_std",
                 ml_model_path: Optional[str] = None):
        """
        初始化掺杂分析器
        
        Args:
            structure_file: 基础结构CIF文件路径
            doping_elements: 掺杂元素列表，如 ["Sr", "Ba", "Al"]
            concentration_range: 掺杂浓度范围 [min, max]
            vasp_work_dir: VASP计算工作目录
            pseudo_dir: 赝势文件目录
            vasp_cmd: VASP执行命令
            ml_model_path: 机器学习模型路径
        """
        try:
            self.base_structure = Structure.from_file(structure_file)
            self.doping_elements = [Element(el) for el in doping_elements]
            self.conc_range = concentration_range
            self.doped_structures = {}
            self.vasp_work_dir = vasp_work_dir
            
            # 创建工作目录
            os.makedirs(vasp_work_dir, exist_ok=True)
            
            # 初始化VASP接口
            self.vasp = VaspInterface(
                work_dir=vasp_work_dir,
                pseudo_dir=pseudo_dir,
                vasp_cmd=vasp_cmd
            )
            
            # 初始化机器学习预测器
            self.ml_predictor = MLDopingSitePredictor(
                model_path=ml_model_path
            )
            
            # 验证输入参数
            if not (0 <= concentration_range[0] <= concentration_range[1] <= 1):
                raise ValueError("浓度范围必须在0到1之间")
                
        except Exception as e:
            raise ValueError(f"初始化失败: {e}")
        
    def analyze_structure(self) -> Dict:
        """分析基础结构的特征"""
        try:
            # 使用VoronoiNN分析配位环境
            voronoi = VoronoiNN()
            sites = self.base_structure.sites
            
            # 分析A位和B位特征
            a_sites = []
            b_sites = []
            
            for idx, site in enumerate(sites):
                # 获取配位环境
                try:
                    neighbors = voronoi.get_nn_info(self.base_structure, idx)
                    coord_num = len(neighbors)
                    
                    # 根据配位数和离子半径判断位点类型
                    if coord_num >= 8:  # A位通常8-12配位
                        a_sites.append(idx)
                    elif 5 <= coord_num <= 6:  # B位通常6配位
                        b_sites.append(idx)
                        
                except Exception as e:
                    print(f"Warning: 分析位点 {idx} 时出错: {e}")
            
            return {
                "lattice_params": dict(zip(
                    ["a", "b", "c", "alpha", "beta", "gamma"],
                    self.base_structure.lattice.parameters
                )),
                "a_sites": len(a_sites),
                "b_sites": len(b_sites),
                "total_sites": len(sites),
                "volume": self.base_structure.volume,
                "density": self.base_structure.density
            }
            
        except Exception as e:
            print(f"Error: 结构分析失败: {e}")
            return {}
    
    def find_doping_sites(self, element: Element) -> List[int]:
        """
        确定掺杂位点
        
        Args:
            element: 掺杂元素
            
        Returns:
            适合掺杂的位点索引列表
        """
        try:
            target_sites = []
            
            # 使用机器学习模型预测每个位点的适合性
            for idx, site in enumerate(self.base_structure):
                try:
                    # 获取预测得分
                    score = self.ml_predictor.predict(
                        self.base_structure,
                        idx,
                        element.symbol
                    )
                    
                    # 如果得分超过阈值，认为是合适的位点
                    if score > 0.7:  # 可以调整阈值
                        target_sites.append((idx, score))
                        
                except Exception as e:
                    print(f"Warning: 预测位点 {idx} 时出错: {e}")
                    
                    # 如果ML预测失败，回退到基于规则的方法
                    voronoi = VoronoiNN()
                    if element.symbol in ["Sr", "Ba"]:  # A位掺杂
                        if site.species_string in ["La", "Li"]:
                            try:
                                neighbors = voronoi.get_nn_info(self.base_structure, idx)
                                if len(neighbors) >= 8:  # A位通常8-12配位
                                    target_sites.append((idx, 0.5))  # 默认得分0.5
                            except Exception as e:
                                print(f"Warning: 分析A位点 {idx} 时出错: {e}")
                                
                    elif element.symbol == "Al":  # B位掺杂
                        if site.species_string in ["Ti", "Nb"]:
                            try:
                                neighbors = voronoi.get_nn_info(self.base_structure, idx)
                                if 5 <= len(neighbors) <= 6:  # B位通常6配位
                                    target_sites.append((idx, 0.5))  # 默认得分0.5
                            except Exception as e:
                                print(f"Warning: 分析B位点 {idx} 时出错: {e}")
            
            # 按得分排序并返回位点索引
            sorted_sites = sorted(target_sites, key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in sorted_sites]
            
        except Exception as e:
            print(f"Error: 寻找掺杂位点失败: {e}")
            return []
    
    def generate_doped_structures(self) -> Dict[str, Structure]:
        """生成掺杂结构"""
        try:
            for element in self.doping_elements:
                # 获取可能的掺杂位点
                doping_sites = self.find_doping_sites(element)
                if not doping_sites:
                    print(f"Warning: 未找到适合 {element.symbol} 的掺杂位点")
                    continue
                
                # 对每个浓度生成结构
                for conc in np.linspace(self.conc_range[0], self.conc_range[1], 3):
                    # 计算需要替换的原子数
                    total_sites = len(self.base_structure)  # 使用总原子数而不是掺杂位点数
                    n_replace = max(1, int(total_sites * conc))
                    
                    # 确保不超过可用的掺杂位点数
                    n_replace = min(n_replace, len(doping_sites))
                    
                    # 选择掺杂位点
                    selected_sites = np.random.choice(
                        doping_sites, 
                        n_replace, 
                        replace=False
                    )
                    
                    # 创建掺杂结构
                    doped = self.base_structure.copy()
                    for site_idx in selected_sites:
                        doped.replace(site_idx, element.symbol)
                    
                    # 保存结构
                    key = f"{element.symbol}_{conc:.3f}"
                    self.doped_structures[key] = doped
                    
            return self.doped_structures
            
        except Exception as e:
            print(f"Error: 生成掺杂结构失败: {e}")
            return {}
    
    def calculate_formation_energy(self, structure: Structure) -> float:
        """
        计算掺杂形成能
        
        Args:
            structure: 掺杂结构
            
        Returns:
            形成能 (eV)
        """
        try:
            # 设置化学势
            chemical_potentials = self._get_chemical_potentials()
            
            # 创建计算目录
            calc_dir = os.path.join(self.vasp_work_dir, "formation_energy")
            os.makedirs(calc_dir, exist_ok=True)
            
            # 使用VASP接口计算形成能
            formation_energy = self.vasp.calculate_formation_energy(
                defect_structure=structure,
                perfect_structure=self.base_structure,
                chemical_potentials=chemical_potentials,
                calc_dir=calc_dir
            )
            
            return formation_energy
            
        except Exception as e:
            print(f"Warning: 计算形成能时出错: {e}")
            return float('inf')
    
    def _get_chemical_potentials(self) -> Dict[str, float]:
        """获取元素的化学势"""
        # 这里可以根据实际情况设置化学势
        # 例如从DFT计算或实验数据获取
        potentials = {
            "Sr": -5.0,
            "Ba": -5.2,
            "Al": -3.8,
            "O": -4.0,
            "Li": -1.9,
            "La": -4.9,
            "Ti": -7.5,
            "Nb": -7.0
        }
        return potentials
    
    def evaluate_properties(self, structures: Dict[str, Structure]) -> Dict:
        """
        评估掺杂结构的性能
        
        Args:
            structures: 掺杂结构字典
            
        Returns:
            性能评估结果
        """
        try:
            results = {}
            for name, structure in structures.items():
                # 计算形成能
                formation_energy = self.calculate_formation_energy(structure)
                
                # 分析结构稳定性
                stability = self._analyze_stability(structure)
                
                # 预测电导率提升
                conductivity_enhancement = self._predict_conductivity(structure)
                
                results[name] = {
                    "formation_energy": formation_energy,
                    "stability": stability,
                    "conductivity_enhancement": conductivity_enhancement
                }
                
            return results
            
        except Exception as e:
            print(f"Error: 性能评估失败: {e}")
            return {}
    
    def _analyze_stability(self, structure: Structure) -> float:
        """
        分析结构稳定性
        
        Returns:
            稳定性得分 (0-1)
        """
        try:
            # 准备VASP计算
            self.vasp.prepare_input_files(
                structure=structure,
                calc_type="relax",
                ISIF=3,  # 允许晶胞和原子位置都优化
                NSW=100  # 最大离子步数
            )
            
            # 运行计算
            if not self.vasp.run_calculation():
                return 0.0
                
            # 获取结果
            results = self.vasp.get_results()
            
            # 分析结构变化
            if not results.get("is_converged", False):
                return 0.0
                
            # 计算体积变化
            initial_volume = structure.volume
            final_volume = results.get("final_structure", structure).volume
            volume_change = abs(final_volume - initial_volume) / initial_volume
            
            # 计算能量变化
            energy_change = abs(results.get("energy", 0) - results.get("initial_energy", 0))
            
            # 综合评分
            stability = 1.0 - (0.5 * volume_change + 0.5 * energy_change / 5.0)
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            print(f"Warning: 稳定性分析失败: {e}")
            return 0.0
    
    def _predict_conductivity(self, structure: Structure) -> float:
        """
        预测电导率提升
        
        Returns:
            预期的电导率提升倍数
        """
        try:
            # 计算电子结构
            self.vasp.prepare_input_files(
                structure=structure,
                calc_type="static",
                LORBIT=11,  # 输出详细的电子结构信息
                ISMEAR=-5   # 使用tetrahedron方法
            )
            
            if not self.vasp.run_calculation():
                return 1.0
                
            results = self.vasp.get_results()
            
            # 分析电子结构
            band_gap = results.get("electronic_structure", {}).get("band_gap", {}).get("energy", 4.0)
            dos_at_fermi = results.get("electronic_structure", {}).get("dos", {}).get("densities", [0])[0]
            
            # 计算稳定性
            stability = self._analyze_stability(structure)
            
            # 估算电导率提升
            # 假设：较小的带隙和较高的DOS有利于电导率
            conductivity_factor = np.exp(-band_gap/2.0) * (1 + dos_at_fermi/10.0)
            enhancement = stability * conductivity_factor
            
            return max(1.0, 1.0 + enhancement)
            
        except Exception as e:
            print(f"Warning: 电导率预测失败: {e}")
            return 1.0
    
    def get_optimization_suggestions(self) -> List[Dict]:
        """
        生成优化建议
        
        Returns:
            优化建议列表
        """
        try:
            suggestions = []
            
            # 分析所有掺杂结构的性能
            properties = self.evaluate_properties(self.doped_structures)
            
            # 根据性能筛选最优配置
            for name, props in properties.items():
                if props["conductivity_enhancement"] > 1.3:  # 提升超过30%
                    element, conc = name.split("_")
                    suggestions.append({
                        "doping_element": element,
                        "concentration": float(conc),
                        "expected_enhancement": props["conductivity_enhancement"],
                        "formation_energy": props["formation_energy"],
                        "stability": props["stability"],
                        "synthesis_conditions": self._get_synthesis_conditions(element)
                    })
            
            return sorted(suggestions, 
                         key=lambda x: x["expected_enhancement"],
                         reverse=True)
                         
        except Exception as e:
            print(f"Error: 生成优化建议失败: {e}")
            return []
    
    def _get_synthesis_conditions(self, element: str) -> Dict:
        """获取合成条件建议"""
        conditions = {
            "Sr": {
                "temperature": 1200,
                "time": 12,
                "atmosphere": "air"
            },
            "Ba": {
                "temperature": 1150,
                "time": 10,
                "atmosphere": "air"
            },
            "Al": {
                "temperature": 1100,
                "time": 8,
                "atmosphere": "air"
            }
        }
        return conditions.get(element, {}) 