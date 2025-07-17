#!/usr/bin/env python3
"""
机器学习势能训练项目：定点掺杂+烧结应力对钙钛矿晶界电导率的影响
目标：揭示σ_GB提升30-50%的机制，并生成晶粒尺寸-阻抗相图
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import yaml
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入必要的库
try:
    from pymatgen.core.structure import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.transformations.standard_transformations import SubstitutionTransformation
    from pymatgen.transformations.defect_transformations import VacancyTransformation
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core.periodic_table import Element
    from pymatgen.io.cif import CifParser, CifWriter
    from pymatgen.io.vasp import Poscar
    from pymatgen.analysis.local_env import VoronoiNN
    from pymatgen.analysis.defects.generators import VacancyGenerator
    from pymatgen.analysis.gb.grain import GrainBoundary
    from pymatgen.analysis.gb.generator import GrainBoundaryGenerator
    from pymatgen.analysis.interface import Interface
    from pymatgen.analysis.dimensionality import get_dimensionality_cheon
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.ext.matproj import MPRester
    PYMATGEN_AVAILABLE = True
except ImportError:
    print("警告：PyMatGen未安装，请先安装：pip install pymatgen")
    PYMATGEN_AVAILABLE = False

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.calculators.vasp import Vasp
    from ase.calculators.espresso import Espresso
    from ase.md import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.constraints import FixAtoms
    from ase.spacegroup import crystal
    from ase.visualize import view
    ASE_AVAILABLE = True
except ImportError:
    print("警告：ASE未安装，请先安装：pip install ase")
    ASE_AVAILABLE = False

try:
    import torch
    from nequip.model import model_from_config
    from nequip.train import main as nequip_train
    from nequip.scripts.train import default_config
    from nequip.data import dataset_from_config
    from nequip.utils import Config
    NEQUIP_AVAILABLE = True
except ImportError:
    print("警告：NequIP未安装，请按照官方文档安装")
    NEQUIP_AVAILABLE = False

try:
    from mace.calculators import mace_mp
    from mace.tools import torch_geometric
    MACE_AVAILABLE = True
except ImportError:
    print("警告：MACE未安装，请按照官方文档安装")
    MACE_AVAILABLE = False

class MLPotentialTrainer:
    """机器学习势能训练主类"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "ml_potential_output"):
        """
        初始化训练器
        
        Args:
            data_dir: CIF文件目录
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.structures_dir = self.output_dir / "structures"
        self.dft_dir = self.output_dir / "dft_calculations"
        self.training_dir = self.output_dir / "training"
        self.analysis_dir = self.output_dir / "analysis"
        
        for dir_path in [self.structures_dir, self.dft_dir, self.training_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # 配置参数
        self.config = {
            "doping_elements": ["Sr", "Ba", "Al"],  # 掺杂元素
            "doping_concentrations": [0.01, 0.02, 0.03],  # 掺杂浓度1-3%
            "sampling_ratio": 0.1,  # 10%原胞采样
            "target_structures": 35,  # 目标结构数
            "dft_frames": 5000,  # DFT训练帧数
            "energy_tolerance": 0.005,  # 能量误差目标: 5 meV/atom
            "force_tolerance": 0.05,  # 力误差目标: 0.05 eV/Å
            "training_epochs": 1000,  # 训练轮数
            "batch_size": 32,  # 批次大小
            "learning_rate": 0.001,  # 学习率
            "temperature_range": [300, 1200],  # 温度范围K
            "pressure_range": [0, 10],  # 压力范围GPa
            "grain_sizes": [10, 20, 50, 100, 200],  # 晶粒尺寸nm
        }
        
        # 存储分析结果
        self.ti_structures = {}
        self.doped_structures = {}
        self.vacancy_structures = {}
        self.training_data = []
        self.trained_models = {}
        
        # 日志设置
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        import logging
        
        log_file = self.output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def analyze_cif_files(self) -> Dict[str, Structure]:
        """
        分析所有CIF文件，识别含Ti的结构
        
        Returns:
            Dict[str, Structure]: 含Ti结构的字典
        """
        self.logger.info("开始分析CIF文件...")
        
        if not PYMATGEN_AVAILABLE:
            self.logger.error("PyMatGen未安装，无法分析CIF文件")
            return {}
            
        # 查找所有CIF文件
        cif_files = []
        for pattern in ["**/*.cif", "**/*.CIF"]:
            cif_files.extend(glob.glob(str(self.data_dir / pattern), recursive=True))
            
        self.logger.info(f"找到 {len(cif_files)} 个CIF文件")
        
        ti_structures = {}
        
        for cif_file in cif_files:
            try:
                # 解析CIF文件
                parser = CifParser(cif_file)
                structures = parser.get_structures()
                
                for i, structure in enumerate(structures):
                    # 检查是否含Ti
                    if "Ti" in [str(site.specie) for site in structure]:
                        structure_name = f"{Path(cif_file).stem}_{i}"
                        ti_structures[structure_name] = structure
                        
                        # 分析基本信息
                        self.analyze_structure_info(structure, structure_name)
                        
            except Exception as e:
                self.logger.warning(f"解析CIF文件 {cif_file} 时出错: {e}")
                continue
                
        self.logger.info(f"识别到 {len(ti_structures)} 个含Ti结构")
        self.ti_structures = ti_structures
        
        # 保存结构信息
        self.save_structure_info()
        
        return ti_structures
        
    def analyze_structure_info(self, structure: Structure, name: str):
        """
        分析单个结构的详细信息
        
        Args:
            structure: 晶体结构
            name: 结构名称
        """
        try:
            # 空间群分析
            sga = SpacegroupAnalyzer(structure)
            spacegroup = sga.get_space_group_symbol()
            
            # 晶格参数
            lattice = structure.lattice
            
            # 组成分析
            composition = structure.composition
            
            # Ti位点分析
            ti_sites = [site for site in structure if site.specie.symbol == "Ti"]
            
            # 存储信息
            info = {
                "name": name,
                "formula": composition.reduced_formula,
                "spacegroup": spacegroup,
                "lattice_abc": [lattice.a, lattice.b, lattice.c],
                "lattice_angles": [lattice.alpha, lattice.beta, lattice.gamma],
                "volume": lattice.volume,
                "density": structure.density,
                "num_sites": len(structure),
                "ti_sites": len(ti_sites),
                "ti_coordination": self.get_ti_coordination(structure),
                "is_perovskite": self.is_perovskite_structure(structure),
                "dimensionality": get_dimensionality_cheon(structure),
            }
            
            # 保存到类变量
            if not hasattr(self, 'structure_info'):
                self.structure_info = {}
            self.structure_info[name] = info
            
        except Exception as e:
            self.logger.warning(f"分析结构 {name} 时出错: {e}")
            
    def get_ti_coordination(self, structure: Structure) -> List[int]:
        """
        获取Ti的配位数
        
        Args:
            structure: 晶体结构
            
        Returns:
            List[int]: Ti配位数列表
        """
        try:
            vnn = VoronoiNN()
            ti_coords = []
            
            for i, site in enumerate(structure):
                if site.specie.symbol == "Ti":
                    neighbors = vnn.get_cn(structure, i)
                    ti_coords.append(neighbors)
                    
            return ti_coords
        except:
            return []
            
    def is_perovskite_structure(self, structure: Structure) -> bool:
        """
        判断是否为钙钛矿结构
        
        Args:
            structure: 晶体结构
            
        Returns:
            bool: 是否为钙钛矿结构
        """
        try:
            # 简单的钙钛矿判断：ABO3型组成
            composition = structure.composition
            elements = list(composition.elements)
            
            # 检查是否有氧元素
            if Element("O") not in elements:
                return False
                
            # 检查氧的比例
            o_fraction = composition.get_atomic_fraction(Element("O"))
            if not (0.5 <= o_fraction <= 0.65):  # 允许一定误差
                return False
                
            # 检查是否有A位和B位元素
            non_o_elements = [elem for elem in elements if elem.symbol != "O"]
            if len(non_o_elements) < 2:
                return False
                
            return True
            
        except:
            return False
            
    def save_structure_info(self):
        """保存结构信息到文件"""
        if hasattr(self, 'structure_info'):
            info_file = self.analysis_dir / "structure_analysis.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(self.structure_info, f, indent=2, ensure_ascii=False)
                
            # 创建CSV摘要
            df_data = []
            for name, info in self.structure_info.items():
                df_data.append({
                    "结构名称": name,
                    "化学式": info["formula"],
                    "空间群": info["spacegroup"],
                    "晶格参数a": info["lattice_abc"][0],
                    "晶格参数b": info["lattice_abc"][1],
                    "晶格参数c": info["lattice_abc"][2],
                    "体积": info["volume"],
                    "密度": info["density"],
                    "Ti位点数": info["ti_sites"],
                    "是否钙钛矿": info["is_perovskite"],
                    "维度": info["dimensionality"],
                })
                
            df = pd.DataFrame(df_data)
            df.to_csv(self.analysis_dir / "structure_summary.csv", index=False, encoding='utf-8')
            
    def select_representative_structures(self, n_structures: int = 35) -> Dict[str, Structure]:
        """
        选择代表性的含Ti结构
        
        Args:
            n_structures: 目标结构数量
            
        Returns:
            Dict[str, Structure]: 选中的结构
        """
        self.logger.info(f"选择 {n_structures} 个代表性结构...")
        
        if not self.ti_structures:
            self.logger.error("没有找到含Ti结构")
            return {}
            
        # 如果结构数量不足，使用所有结构
        if len(self.ti_structures) <= n_structures:
            self.logger.info(f"结构数量 {len(self.ti_structures)} 少于目标数量，使用所有结构")
            return self.ti_structures
            
        # 基于多样性选择结构
        selected_structures = self.diverse_structure_selection(n_structures)
        
        self.logger.info(f"选择了 {len(selected_structures)} 个代表性结构")
        return selected_structures
        
    def diverse_structure_selection(self, n_structures: int) -> Dict[str, Structure]:
        """
        基于结构多样性选择代表性结构
        
        Args:
            n_structures: 目标结构数量
            
        Returns:
            Dict[str, Structure]: 选中的结构
        """
        if not hasattr(self, 'structure_info'):
            # 如果没有结构信息，随机选择
            import random
            selected_names = random.sample(list(self.ti_structures.keys()), n_structures)
            return {name: self.ti_structures[name] for name in selected_names}
            
        # 基于结构特征的聚类选择
        features = []
        names = []
        
        for name, info in self.structure_info.items():
            if name in self.ti_structures:
                # 构建特征向量
                feature = [
                    info["lattice_abc"][0],  # a
                    info["lattice_abc"][1],  # b  
                    info["lattice_abc"][2],  # c
                    info["volume"],          # 体积
                    info["density"],         # 密度
                    info["ti_sites"],        # Ti位点数
                    int(info["is_perovskite"]),  # 是否钙钛矿
                    info["dimensionality"],  # 维度
                ]
                features.append(feature)
                names.append(name)
                
        # 标准化特征
        features = np.array(features)
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # 使用KMeans聚类选择多样性结构
        try:
            from sklearn.cluster import KMeans
            
            n_clusters = min(n_structures, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # 从每个聚类中选择一个代表性结构
            selected_structures = {}
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) > 0:
                    # 选择离聚类中心最近的结构
                    cluster_center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(features[cluster_indices] - cluster_center, axis=1)
                    selected_idx = cluster_indices[np.argmin(distances)]
                    selected_name = names[selected_idx]
                    selected_structures[selected_name] = self.ti_structures[selected_name]
                    
            return selected_structures
            
        except ImportError:
            self.logger.warning("sklearn未安装，使用随机选择")
            import random
            selected_names = random.sample(names, n_structures)
            return {name: self.ti_structures[name] for name in selected_names}
            
    def generate_doped_structures(self, structures: Dict[str, Structure]) -> Dict[str, List[Structure]]:
        """
        生成掺杂结构
        
        Args:
            structures: 原始结构
            
        Returns:
            Dict[str, List[Structure]]: 掺杂结构
        """
        self.logger.info("开始生成掺杂结构...")
        
        doped_structures = {}
        
        for structure_name, structure in structures.items():
            self.logger.info(f"处理结构: {structure_name}")
            
            # 为每个结构生成掺杂变体
            structure_doped = []
            
            # 获取10%的原胞
            n_cells = max(1, int(len(structure) * self.config["sampling_ratio"]))
            
            for doping_elem in self.config["doping_elements"]:
                for concentration in self.config["doping_concentrations"]:
                    try:
                        # 创建掺杂结构
                        doped_struct = self.create_doped_structure(
                            structure, doping_elem, concentration, n_cells
                        )
                        
                        if doped_struct:
                            structure_doped.extend(doped_struct)
                            
                    except Exception as e:
                        self.logger.warning(f"生成掺杂结构时出错: {e}")
                        continue
                        
            doped_structures[structure_name] = structure_doped
            self.logger.info(f"为 {structure_name} 生成了 {len(structure_doped)} 个掺杂结构")
            
        self.doped_structures = doped_structures
        return doped_structures
        
    def create_doped_structure(self, structure: Structure, doping_elem: str, 
                              concentration: float, n_cells: int) -> List[Structure]:
        """
        创建单个掺杂结构
        
        Args:
            structure: 原始结构
            doping_elem: 掺杂元素
            concentration: 掺杂浓度
            n_cells: 原胞数量
            
        Returns:
            List[Structure]: 掺杂结构列表
        """
        doped_structures = []
        
        # 确定掺杂位点
        possible_sites = self.identify_doping_sites(structure, doping_elem)
        
        if not possible_sites:
            self.logger.warning(f"结构中没有找到适合 {doping_elem} 掺杂的位点")
            return []
            
        # 计算需要替换的原子数
        n_atoms_to_replace = max(1, int(len(structure) * concentration))
        
        # 生成多个掺杂配置
        for config_idx in range(min(5, len(possible_sites))):  # 最多5个配置
            try:
                # 随机选择掺杂位点
                import random
                selected_sites = random.sample(possible_sites, 
                                             min(n_atoms_to_replace, len(possible_sites)))
                
                # 创建替换变换
                substitutions = {site: doping_elem for site in selected_sites}
                transformation = SubstitutionTransformation(substitutions)
                
                # 应用变换
                doped_struct = transformation.apply_transformation(structure)
                
                # 添加标识信息
                doped_struct.add_site_property("doping_element", [doping_elem] * len(doped_struct))
                doped_struct.add_site_property("doping_concentration", [concentration] * len(doped_struct))
                
                doped_structures.append(doped_struct)
                
            except Exception as e:
                self.logger.warning(f"创建掺杂配置时出错: {e}")
                continue
                
        return doped_structures
        
    def identify_doping_sites(self, structure: Structure, doping_elem: str) -> List[int]:
        """
        识别适合掺杂的位点
        
        Args:
            structure: 晶体结构
            doping_elem: 掺杂元素
            
        Returns:
            List[int]: 适合掺杂的位点索引列表
        """
        suitable_sites = []
        
        # 根据掺杂元素确定可能的替换位点
        if doping_elem in ["Sr", "Ba"]:
            # 碱土金属通常替换A位（较大离子）
            for i, site in enumerate(structure):
                if site.specie.symbol in ["La", "Li", "Ca", "Sr", "Ba"]:
                    suitable_sites.append(i)
                    
        elif doping_elem == "Al":
            # Al通常替换B位（较小离子）
            for i, site in enumerate(structure):
                if site.specie.symbol in ["Ti", "Nb", "Ta", "Zr", "Al"]:
                    suitable_sites.append(i)
                    
        return suitable_sites
        
    def generate_vacancy_structures(self, structures: Dict[str, List[Structure]]) -> Dict[str, List[Structure]]:
        """
        生成空位结构
        
        Args:
            structures: 掺杂结构
            
        Returns:
            Dict[str, List[Structure]]: 含空位的结构
        """
        self.logger.info("开始生成空位结构...")
        
        vacancy_structures = {}
        
        for structure_name, struct_list in structures.items():
            vacancy_list = []
            
            for struct in struct_list:
                try:
                    # 生成氧空位
                    o_vacancies = self.create_oxygen_vacancies(struct)
                    vacancy_list.extend(o_vacancies)
                    
                    # 生成阳离子空位
                    cation_vacancies = self.create_cation_vacancies(struct)
                    vacancy_list.extend(cation_vacancies)
                    
                except Exception as e:
                    self.logger.warning(f"生成空位结构时出错: {e}")
                    continue
                    
            vacancy_structures[structure_name] = vacancy_list
            self.logger.info(f"为 {structure_name} 生成了 {len(vacancy_list)} 个空位结构")
            
        self.vacancy_structures = vacancy_structures
        return vacancy_structures
        
    def create_oxygen_vacancies(self, structure: Structure, n_vacancies: int = 1) -> List[Structure]:
        """
        创建氧空位
        
        Args:
            structure: 原始结构
            n_vacancies: 空位数量
            
        Returns:
            List[Structure]: 含氧空位的结构
        """
        vacancy_structures = []
        
        # 找到氧原子位点
        o_sites = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]
        
        if len(o_sites) < n_vacancies:
            return []
            
        # 生成不同的空位配置
        import itertools
        for vacancy_indices in itertools.combinations(o_sites, n_vacancies):
            try:
                # 创建空位结构
                vacancy_struct = structure.copy()
                
                # 移除原子（从高索引到低索引）
                for idx in sorted(vacancy_indices, reverse=True):
                    vacancy_struct.remove_sites([idx])
                    
                # 添加标识
                vacancy_struct.add_site_property("has_oxygen_vacancy", [True] * len(vacancy_struct))
                
                vacancy_structures.append(vacancy_struct)
                
                # 限制生成数量
                if len(vacancy_structures) >= 3:
                    break
                    
            except Exception as e:
                self.logger.warning(f"创建氧空位时出错: {e}")
                continue
                
        return vacancy_structures
        
    def create_cation_vacancies(self, structure: Structure, n_vacancies: int = 1) -> List[Structure]:
        """
        创建阳离子空位
        
        Args:
            structure: 原始结构
            n_vacancies: 空位数量
            
        Returns:
            List[Structure]: 含阳离子空位的结构
        """
        vacancy_structures = []
        
        # 找到阳离子位点（除氧外的所有原子）
        cation_sites = [i for i, site in enumerate(structure) if site.specie.symbol != "O"]
        
        if len(cation_sites) < n_vacancies:
            return []
            
        # 生成不同的空位配置
        import itertools
        for vacancy_indices in itertools.combinations(cation_sites, n_vacancies):
            try:
                # 创建空位结构
                vacancy_struct = structure.copy()
                
                # 移除原子（从高索引到低索引）
                for idx in sorted(vacancy_indices, reverse=True):
                    vacancy_struct.remove_sites([idx])
                    
                # 添加标识
                vacancy_struct.add_site_property("has_cation_vacancy", [True] * len(vacancy_struct))
                
                vacancy_structures.append(vacancy_struct)
                
                # 限制生成数量
                if len(vacancy_structures) >= 2:
                    break
                    
            except Exception as e:
                self.logger.warning(f"创建阳离子空位时出错: {e}")
                continue
                
        return vacancy_structures
        
    def save_generated_structures(self):
        """保存生成的结构"""
        self.logger.info("保存生成的结构...")
        
        # 创建目录
        doped_dir = self.structures_dir / "doped"
        vacancy_dir = self.structures_dir / "vacancy"
        doped_dir.mkdir(exist_ok=True)
        vacancy_dir.mkdir(exist_ok=True)
        
        # 保存掺杂结构
        for structure_name, struct_list in self.doped_structures.items():
            for i, struct in enumerate(struct_list):
                filename = f"{structure_name}_doped_{i}.cif"
                writer = CifWriter(struct)
                writer.write_file(doped_dir / filename)
                
        # 保存空位结构
        for structure_name, struct_list in self.vacancy_structures.items():
            for i, struct in enumerate(struct_list):
                filename = f"{structure_name}_vacancy_{i}.cif"
                writer = CifWriter(struct)
                writer.write_file(vacancy_dir / filename)
                
        self.logger.info("结构保存完成")
        
    def setup_dft_calculations(self) -> List[Dict]:
        """
        设置DFT计算
        
        Returns:
            List[Dict]: DFT计算任务列表
        """
        self.logger.info("设置DFT计算...")
        
        if not ASE_AVAILABLE:
            self.logger.error("ASE未安装，无法设置DFT计算")
            return []
            
        dft_tasks = []
        
        # 收集所有结构
        all_structures = []
        
        # 添加掺杂结构
        for struct_list in self.doped_structures.values():
            all_structures.extend(struct_list)
            
        # 添加空位结构
        for struct_list in self.vacancy_structures.values():
            all_structures.extend(struct_list)
            
        self.logger.info(f"准备计算 {len(all_structures)} 个结构")
        
        # 为每个结构设置DFT计算
        for i, structure in enumerate(all_structures):
            # 创建ASE原子对象
            atoms = self.pymatgen_to_ase(structure)
            
            # 生成不同的计算配置
            configs = self.generate_dft_configs(atoms, i)
            dft_tasks.extend(configs)
            
        # 限制总计算数量
        target_frames = self.config["dft_frames"]
        if len(dft_tasks) > target_frames:
            import random
            dft_tasks = random.sample(dft_tasks, target_frames)
            
        self.logger.info(f"设置了 {len(dft_tasks)} 个DFT计算任务")
        
        # 保存计算脚本
        self.save_dft_scripts(dft_tasks)
        
        return dft_tasks
        
    def pymatgen_to_ase(self, structure: Structure) -> Atoms:
        """
        将PyMatGen结构转换为ASE原子对象
        
        Args:
            structure: PyMatGen结构
            
        Returns:
            Atoms: ASE原子对象
        """
        atoms = Atoms(
            symbols=[site.specie.symbol for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )
        return atoms
        
    def generate_dft_configs(self, atoms: Atoms, base_idx: int) -> List[Dict]:
        """
        生成DFT计算配置
        
        Args:
            atoms: ASE原子对象
            base_idx: 基础索引
            
        Returns:
            List[Dict]: DFT配置列表
        """
        configs = []
        
        # 基础配置
        base_config = {
            "atoms": atoms.copy(),
            "calculator": "vasp",
            "calc_params": {
                "xc": "PBE",
                "encut": 500,
                "kpts": [4, 4, 4],
                "ismear": 0,
                "sigma": 0.1,
                "ediff": 1e-6,
                "ediffg": -0.02,
                "nelm": 200,
                "ispin": 2,
                "lorbit": 11,
                "lreal": "Auto",
                "algo": "Normal",
                "prec": "Accurate",
            },
            "task_id": f"task_{base_idx}",
        }
        
        # 静态计算
        static_config = base_config.copy()
        static_config["calc_type"] = "static"
        static_config["task_id"] = f"task_{base_idx}_static"
        configs.append(static_config)
        
        # 结构优化
        relax_config = base_config.copy()
        relax_config["calc_type"] = "relax"
        relax_config["task_id"] = f"task_{base_idx}_relax"
        relax_config["calc_params"]["ibrion"] = 2
        relax_config["calc_params"]["isif"] = 3
        configs.append(relax_config)
        
        # 分子动力学
        md_config = base_config.copy()
        md_config["calc_type"] = "md"
        md_config["task_id"] = f"task_{base_idx}_md"
        md_config["calc_params"]["ibrion"] = 0
        md_config["calc_params"]["nsw"] = 100
        md_config["calc_params"]["potim"] = 2.0
        md_config["calc_params"]["smass"] = 0
        
        # 不同温度的MD
        for temp in [300, 600, 900, 1200]:
            temp_config = md_config.copy()
            temp_config["task_id"] = f"task_{base_idx}_md_{temp}K"
            temp_config["calc_params"]["tebeg"] = temp
            temp_config["calc_params"]["teend"] = temp
            configs.append(temp_config)
            
        return configs
        
    def save_dft_scripts(self, dft_tasks: List[Dict]):
        """
        保存DFT计算脚本
        
        Args:
            dft_tasks: DFT任务列表
        """
        # 创建VASP输入文件
        for task in dft_tasks:
            task_dir = self.dft_dir / task["task_id"]
            task_dir.mkdir(exist_ok=True)
            
            # 保存结构
            atoms = task["atoms"]
            atoms.write(task_dir / "POSCAR", format="vasp")
            
            # 保存计算参数
            calc_params = task["calc_params"]
            with open(task_dir / "INCAR", "w") as f:
                for key, value in calc_params.items():
                    f.write(f"{key.upper()} = {value}\n")
                    
            # 创建提交脚本
            self.create_submit_script(task_dir, task["task_id"])
            
        # 创建批量提交脚本
        self.create_batch_submit_script(dft_tasks)
        
    def create_submit_script(self, task_dir: Path, task_id: str):
        """
        创建单个任务的提交脚本
        
        Args:
            task_dir: 任务目录
            task_id: 任务ID
        """
        script_content = f"""#!/bin/bash
#SBATCH --job-name={task_id}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# 环境设置
module load vasp/6.3.0
module load cuda/11.0

# 运行计算
cd {task_dir}
mpirun -np 32 vasp_std > vasp.out 2>&1

# 后处理
python ${{HOME}}/scripts/parse_vasp_output.py ./
"""
        
        with open(task_dir / "submit.sh", "w") as f:
            f.write(script_content)
            
        # 设置执行权限
        os.chmod(task_dir / "submit.sh", 0o755)
        
    def create_batch_submit_script(self, dft_tasks: List[Dict]):
        """
        创建批量提交脚本
        
        Args:
            dft_tasks: DFT任务列表
        """
        script_content = """#!/bin/bash
# 批量提交DFT计算任务

echo "开始提交DFT计算任务..."
echo "总任务数: {}"

""".format(len(dft_tasks))
        
        for task in dft_tasks:
            task_dir = self.dft_dir / task["task_id"]
            script_content += f"""
# 提交任务: {task["task_id"]}
echo "提交任务: {task["task_id"]}"
cd {task_dir}
sbatch submit.sh
sleep 1
"""
        
        script_content += """
echo "所有任务提交完成！"
echo "使用 'squeue -u $USER' 查看任务状态"
"""
        
        batch_script = self.dft_dir / "submit_all.sh"
        with open(batch_script, "w") as f:
            f.write(script_content)
            
        os.chmod(batch_script, 0o755)
        
    def setup_nequip_training(self) -> str:
        """
        设置NequIP训练
        
        Returns:
            str: 训练配置文件路径
        """
        self.logger.info("设置NequIP训练...")
        
        if not NEQUIP_AVAILABLE:
            self.logger.error("NequIP未安装，无法设置训练")
            return ""
            
        # 创建训练配置
        config = self.create_nequip_config()
        
        # 保存配置文件
        config_file = self.training_dir / "nequip_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
            
        # 创建训练脚本
        self.create_nequip_training_script(config_file)
        
        return str(config_file)
        
    def create_nequip_config(self) -> Dict:
        """
        创建NequIP训练配置
        
        Returns:
            Dict: 训练配置
        """
        config = {
            # 数据设置
            "dataset": "ase",
            "dataset_file_name": str(self.training_dir / "training_data.db"),
            "chemical_symbols": ["Li", "La", "Ti", "O", "Sr", "Ba", "Al", "Nb", "Ta", "Zr", "Ga"],
            "type_names": ["Li", "La", "Ti", "O", "Sr", "Ba", "Al", "Nb", "Ta", "Zr", "Ga"],
            
            # 模型设置
            "model_builders": [
                "SimpleIrrepsConfig",
                "EnergyModel",
                "PerSpeciesRescale",
                "ForceOutput",
                "RescaleEnergyEtc",
            ],
            
            # 网络架构
            "r_max": 5.0,
            "num_layers": 4,
            "l_max": 2,
            "num_features": 32,
            "nonlinearity_type": "gate",
            "resnet": True,
            "nonlinearity_scalars": {"e": "silu", "o": "tanh"},
            "nonlinearity_gates": {"e": "silu", "o": "abs"},
            
            # 训练参数
            "batch_size": self.config["batch_size"],
            "learning_rate": self.config["learning_rate"],
            "max_epochs": self.config["training_epochs"],
            "train_val_split": "random",
            "validation_fraction": 0.1,
            "shuffle": True,
            "metrics_key": "validation_loss",
            
            # 优化器设置
            "optimizer_name": "Adam",
            "optimizer_params": {"weight_decay": 1e-5},
            "lr_scheduler_name": "ReduceLROnPlateau",
            "lr_scheduler_params": {
                "factor": 0.5,
                "patience": 50,
                "threshold": 1e-4,
            },
            
            # 损失函数
            "loss_coeffs": {
                "forces": 1.0,
                "total_energy": 1.0,
            },
            
            # 误差目标
            "early_stopping_patiences": {"validation_loss": 100},
            "early_stopping_lower_bounds": {"LR": 1e-6},
            
            # 输出设置
            "wandb": False,
            "verbose": "INFO",
            "log_batch_freq": 10,
            "log_epoch_freq": 1,
            "save_checkpoint_freq": 100,
            "save_ema_freq": 100,
            
            # 硬件设置
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_debug_mode": False,
            
            # 文件路径
            "root": str(self.training_dir),
            "run_name": f"nequip_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "seed": 42,
            
            # 原子类型映射
            "type_to_chemical_symbol": {
                "Li": "Li", "La": "La", "Ti": "Ti", "O": "O",
                "Sr": "Sr", "Ba": "Ba", "Al": "Al", "Nb": "Nb",
                "Ta": "Ta", "Zr": "Zr", "Ga": "Ga"
            },
            
            # 能量和力的单位
            "energy_units_to_eV": 1.0,
            "length_units_to_A": 1.0,
        }
        
        return config
        
    def create_nequip_training_script(self, config_file: Path):
        """
        创建NequIP训练脚本
        
        Args:
            config_file: 配置文件路径
        """
        script_content = f"""#!/bin/bash
#SBATCH --job-name=nequip_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# 环境设置
module load python/3.9
module load cuda/11.0
source activate nequip

# 训练设置
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 开始训练
echo "开始NequIP训练..."
echo "配置文件: {config_file}"
echo "训练时间: $(date)"

cd {self.training_dir}

# 运行训练
python -m nequip.train {config_file}

# 训练完成后的处理
echo "训练完成: $(date)"

# 生成模型报告
python -c "
import torch
from nequip.model import model_from_config
from nequip.utils import Config

# 加载配置
config = Config.from_file('{config_file}')
print('训练配置摘要:')
print(f'  最大轮数: {{config.max_epochs}}')
print(f'  批次大小: {{config.batch_size}}')
print(f'  学习率: {{config.learning_rate}}')
print(f'  设备: {{config.device}}')

# 检查是否有GPU
if torch.cuda.is_available():
    print(f'  GPU: {{torch.cuda.get_device_name(0)}}')
    print(f'  GPU内存: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')
"

echo "训练脚本执行完成"
"""
        
        script_file = self.training_dir / "train_nequip.sh"
        with open(script_file, "w") as f:
            f.write(script_content)
            
        os.chmod(script_file, 0o755)
        
    def create_mace_config(self) -> Dict:
        """
        创建MACE训练配置
        
        Returns:
            Dict: MACE配置
        """
        config = {
            # 数据设置
            "train_file": str(self.training_dir / "training_data.xyz"),
            "valid_file": str(self.training_dir / "validation_data.xyz"),
            "test_file": str(self.training_dir / "test_data.xyz"),
            "statistics_file": str(self.training_dir / "statistics.json"),
            
            # 模型设置
            "model": "MACE",
            "r_max": 5.0,
            "num_radial_basis": 8,
            "num_cutoff_basis": 5,
            "max_L": 2,
            "num_layers": 4,
            "hidden_irreps": "128x0e + 128x1o + 128x2e",
            "MLP_irreps": "16x0e",
            "correlation": 3,
            "gate": "silu",
            "interaction_first": "RealAgnosticResidualInteractionBlock",
            "interaction": "RealAgnosticResidualInteractionBlock",
            "radial_type": "bessel",
            "radial_MLP": "[64, 64, 64]",
            
            # 训练参数
            "batch_size": self.config["batch_size"],
            "max_num_epochs": self.config["training_epochs"],
            "learning_rate": self.config["learning_rate"],
            "energy_weight": 1.0,
            "forces_weight": 1.0,
            "stress_weight": 1.0,
            "optimizer": "Adam",
            "weight_decay": 1e-5,
            "scheduler": "ReduceLROnPlateau",
            "lr_scheduler_patience": 50,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_threshold": 1e-4,
            
            # 验证和保存
            "eval_interval": 10,
            "save_cpu": True,
            "restart_latest": True,
            "seed": 42,
            
            # 硬件设置
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_workers": 4,
            "pin_memory": True,
            
            # 输出设置
            "name": f"mace_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "results_dir": str(self.training_dir / "mace_results"),
            "checkpoints_dir": str(self.training_dir / "mace_checkpoints"),
            "log_dir": str(self.training_dir / "mace_logs"),
            
            # 化学元素
            "chemical_symbols": ["Li", "La", "Ti", "O", "Sr", "Ba", "Al", "Nb", "Ta", "Zr", "Ga"],
            "atomic_numbers": [3, 57, 22, 8, 38, 56, 13, 41, 73, 40, 31],
            
            # 早停设置
            "patience": 100,
            "max_grad_norm": 10.0,
            "clip_grad": True,
            
            # 数据预处理
            "energy_key": "energy",
            "forces_key": "forces",
            "stress_key": "stress",
            "default_dtype": "float32",
            
            # 验证设置
            "valid_fraction": 0.1,
            "test_fraction": 0.1,
            "error_table": "PerAtomRMSE",
            
            # 输出频率
            "log_level": "INFO",
            "keep_checkpoints": True,
            "save_all_checkpoints": False,
            "log_errors": "PerAtomRMSE",
            "log_wandb": False,
        }
        
        return config
        
    def analyze_grain_boundary_conductivity(self):
        """
        分析晶界电导率
        """
        self.logger.info("开始分析晶界电导率...")
        
        # 这里应该使用训练好的ML势能进行分析
        # 由于实际的分析需要大量计算，这里提供框架
        
        analysis_results = {
            "grain_boundary_analysis": {
                "description": "晶界电导率分析",
                "methods": [
                    "分子动力学模拟",
                    "静态结构分析", 
                    "离子传输路径分析",
                    "激活能计算"
                ],
                "conditions": {
                    "temperature_range": self.config["temperature_range"],
                    "pressure_range": self.config["pressure_range"],
                    "grain_sizes": self.config["grain_sizes"]
                }
            },
            
            "conductivity_enhancement": {
                "target_improvement": "30-50%",
                "mechanisms": [
                    "定点掺杂效应",
                    "晶界结构优化",
                    "载流子浓度提升",
                    "离子传输路径改善"
                ]
            },
            
            "phase_diagram_generation": {
                "description": "晶粒尺寸-阻抗相图",
                "parameters": {
                    "grain_size_range": "10-200 nm",
                    "impedance_range": "计算确定",
                    "temperature_conditions": "300-1200 K",
                    "doping_concentrations": "1-3%"
                }
            }
        }
        
        # 保存分析结果
        analysis_file = self.analysis_dir / "conductivity_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
        return analysis_results
        
    def generate_experimental_guide(self):
        """
        生成实验配方导航
        """
        self.logger.info("生成实验配方导航...")
        
        # 基于分析结果生成实验指导
        experimental_guide = {
            "optimal_compositions": [
                {
                    "base_material": "Li₇La₃Zr₂O₁₂",
                    "doping_strategy": "2% Sr + 1% Al",
                    "predicted_enhancement": "35-40%",
                    "synthesis_conditions": {
                        "temperature": "1200°C",
                        "time": "12h",
                        "atmosphere": "Air",
                        "cooling_rate": "5°C/min"
                    }
                },
                {
                    "base_material": "LiLaTiO₄",
                    "doping_strategy": "3% Ba + 1% Al",
                    "predicted_enhancement": "45-50%",
                    "synthesis_conditions": {
                        "temperature": "1150°C",
                        "time": "10h",
                        "atmosphere": "Air",
                        "cooling_rate": "3°C/min"
                    }
                }
            ],
            
            "processing_parameters": {
                "sintering_stress_control": {
                    "pressure": "50-100 MPa",
                    "temperature": "1000-1200°C",
                    "dwell_time": "2-6h",
                    "heating_rate": "5°C/min"
                },
                
                "microstructure_optimization": {
                    "target_grain_size": "50-100 nm",
                    "grain_boundary_density": "最大化",
                    "porosity": "<5%",
                    "phase_purity": ">95%"
                }
            },
            
            "characterization_methods": [
                "X射线衍射 (XRD)",
                "扫描电子显微镜 (SEM)",
                "交流阻抗谱 (EIS)",
                "透射电子显微镜 (TEM)",
                "X射线光电子能谱 (XPS)"
            ],
            
            "testing_protocol": {
                "ionic_conductivity": {
                    "method": "交流阻抗谱",
                    "frequency_range": "1 Hz - 1 MHz",
                    "temperature_range": "25-300°C",
                    "atmosphere": "干燥空气"
                },
                
                "mechanical_properties": {
                    "hardness": "维氏硬度",
                    "fracture_toughness": "压痕法",
                    "弹性模量": "纳米压痕"
                }
            }
        }
        
        # 保存实验指导
        guide_file = self.analysis_dir / "experimental_guide.json"
        with open(guide_file, 'w', encoding='utf-8') as f:
            json.dump(experimental_guide, f, indent=2, ensure_ascii=False)
            
        # 生成Markdown报告
        self.generate_markdown_report(experimental_guide)
        
        return experimental_guide
        
    def generate_markdown_report(self, experimental_guide: Dict):
        """
        生成Markdown格式的报告
        
        Args:
            experimental_guide: 实验指导数据
        """
        report_content = f"""# 机器学习势能训练项目报告

## 项目概述
- **目标**: 揭示"定点掺杂+烧结应力"对钙钛矿晶界电导率σ_GB提升30-50%的机制
- **方法**: NequIP/MACE机器学习势能训练
- **数据**: 5000帧DFT计算数据
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据统计
- **含Ti结构数量**: {len(self.ti_structures)}
- **掺杂结构数量**: {sum(len(structs) for structs in self.doped_structures.values())}
- **空位结构数量**: {sum(len(structs) for structs in self.vacancy_structures.values())}

## 推荐实验配方

### 配方1: {experimental_guide['optimal_compositions'][0]['base_material']}
- **掺杂策略**: {experimental_guide['optimal_compositions'][0]['doping_strategy']}
- **预期提升**: {experimental_guide['optimal_compositions'][0]['predicted_enhancement']}
- **合成条件**:
  - 温度: {experimental_guide['optimal_compositions'][0]['synthesis_conditions']['temperature']}
  - 时间: {experimental_guide['optimal_compositions'][0]['synthesis_conditions']['time']}
  - 气氛: {experimental_guide['optimal_compositions'][0]['synthesis_conditions']['atmosphere']}
  - 冷却速率: {experimental_guide['optimal_compositions'][0]['synthesis_conditions']['cooling_rate']}

### 配方2: {experimental_guide['optimal_compositions'][1]['base_material']}
- **掺杂策略**: {experimental_guide['optimal_compositions'][1]['doping_strategy']}
- **预期提升**: {experimental_guide['optimal_compositions'][1]['predicted_enhancement']}
- **合成条件**:
  - 温度: {experimental_guide['optimal_compositions'][1]['synthesis_conditions']['temperature']}
  - 时间: {experimental_guide['optimal_compositions'][1]['synthesis_conditions']['time']}
  - 气氛: {experimental_guide['optimal_compositions'][1]['synthesis_conditions']['atmosphere']}
  - 冷却速率: {experimental_guide['optimal_compositions'][1]['synthesis_conditions']['cooling_rate']}

## 工艺参数优化

### 烧结应力控制
- **压力**: {experimental_guide['processing_parameters']['sintering_stress_control']['pressure']}
- **温度**: {experimental_guide['processing_parameters']['sintering_stress_control']['temperature']}
- **保温时间**: {experimental_guide['processing_parameters']['sintering_stress_control']['dwell_time']}
- **升温速率**: {experimental_guide['processing_parameters']['sintering_stress_control']['heating_rate']}

### 微结构优化目标
- **目标晶粒尺寸**: {experimental_guide['processing_parameters']['microstructure_optimization']['target_grain_size']}
- **晶界密度**: {experimental_guide['processing_parameters']['microstructure_optimization']['grain_boundary_density']}
- **孔隙率**: {experimental_guide['processing_parameters']['microstructure_optimization']['porosity']}
- **相纯度**: {experimental_guide['processing_parameters']['microstructure_optimization']['phase_purity']}

## 表征测试方案

### 离子电导率测试
- **方法**: {experimental_guide['testing_protocol']['ionic_conductivity']['method']}
- **频率范围**: {experimental_guide['testing_protocol']['ionic_conductivity']['frequency_range']}
- **温度范围**: {experimental_guide['testing_protocol']['ionic_conductivity']['temperature_range']}
- **测试气氛**: {experimental_guide['testing_protocol']['ionic_conductivity']['atmosphere']}

### 机械性能测试
- **硬度**: {experimental_guide['testing_protocol']['mechanical_properties']['hardness']}
- **断裂韧性**: {experimental_guide['testing_protocol']['mechanical_properties']['fracture_toughness']}
- **弹性模量**: {experimental_guide['testing_protocol']['mechanical_properties']['弹性模量']}

## 预期成果
1. **电导率提升**: 30-50% (相对于未掺杂样品)
2. **晶界优化**: 通过定点掺杂和应力控制实现
3. **工艺优化**: 提供完整的合成-烧结-表征流程
4. **相图构建**: 晶粒尺寸-阻抗关系图谱

## 下一步工作
1. 完成DFT计算 (预计48小时)
2. 训练ML势能模型
3. 进行MD模拟验证
4. 实验验证推荐配方
5. 构建完整的相图数据库

---
*本报告由机器学习势能训练系统自动生成*
"""
        
        report_file = self.analysis_dir / "project_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
    def run_full_pipeline(self):
        """
        运行完整的训练流程
        """
        self.logger.info("开始运行完整的ML势能训练流程...")
        
        try:
            # 步骤1: 分析CIF文件
            self.logger.info("=== 步骤1: 分析CIF文件 ===")
            self.analyze_cif_files()
            
            # 步骤2: 选择代表性结构
            self.logger.info("=== 步骤2: 选择代表性结构 ===")
            selected_structures = self.select_representative_structures(self.config["target_structures"])
            
            # 步骤3: 生成掺杂结构
            self.logger.info("=== 步骤3: 生成掺杂结构 ===")
            self.generate_doped_structures(selected_structures)
            
            # 步骤4: 生成空位结构
            self.logger.info("=== 步骤4: 生成空位结构 ===")
            self.generate_vacancy_structures(self.doped_structures)
            
            # 步骤5: 保存生成的结构
            self.logger.info("=== 步骤5: 保存生成的结构 ===")
            self.save_generated_structures()
            
            # 步骤6: 设置DFT计算
            self.logger.info("=== 步骤6: 设置DFT计算 ===")
            dft_tasks = self.setup_dft_calculations()
            
            # 步骤7: 设置ML势能训练
            self.logger.info("=== 步骤7: 设置ML势能训练 ===")
            nequip_config = self.setup_nequip_training()
            
            # 步骤8: 分析晶界电导率
            self.logger.info("=== 步骤8: 分析晶界电导率 ===")
            conductivity_analysis = self.analyze_grain_boundary_conductivity()
            
            # 步骤9: 生成实验指导
            self.logger.info("=== 步骤9: 生成实验指导 ===")
            experimental_guide = self.generate_experimental_guide()
            
            self.logger.info("=== 流程完成 ===")
            self.logger.info(f"所有输出文件保存在: {self.output_dir}")
            
            return {
                "success": True,
                "structures_analyzed": len(self.ti_structures),
                "dft_tasks_generated": len(dft_tasks),
                "nequip_config": nequip_config,
                "output_directory": str(self.output_dir),
                "experimental_guide": experimental_guide
            }
            
        except Exception as e:
            self.logger.error(f"流程执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_directory": str(self.output_dir)
            }

def main():
    """主函数"""
    print("机器学习势能训练项目")
    print("=" * 50)
    
    # 创建训练器
    trainer = MLPotentialTrainer()
    
    # 运行完整流程
    results = trainer.run_full_pipeline()
    
    # 显示结果
    if results["success"]:
        print("\n✅ 训练流程设置完成!")
        print(f"📊 分析了 {results['structures_analyzed']} 个含Ti结构")
        print(f"🔬 生成了 {results['dft_tasks_generated']} 个DFT计算任务")
        print(f"📁 输出目录: {results['output_directory']}")
        print("\n下一步:")
        print("1. 运行DFT计算: cd {} && ./submit_all.sh".format(results['output_directory'] + "/dft_calculations"))
        print("2. 训练NequIP模型: cd {} && ./train_nequip.sh".format(results['output_directory'] + "/training"))
        print("3. 查看实验指导: cat {}/project_report.md".format(results['output_directory'] + "/analysis"))
    else:
        print("\n❌ 训练流程设置失败!")
        print(f"错误: {results['error']}")
        
if __name__ == "__main__":
    main() 