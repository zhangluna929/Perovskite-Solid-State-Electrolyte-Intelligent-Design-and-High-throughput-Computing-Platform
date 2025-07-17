"""
迁移势垒分析模块
用于计算和分析离子在钙钛矿材料中的迁移势垒
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pymatgen.core import Structure
from pymatgen.analysis.path_finder import NEBPathfinder
from pymatgen.analysis.diffusion.neb.full_path_mapper import FullPathMapper
from pymatgen.analysis.transition_state import NEBAnalysis
from pymatgen.io.vasp.outputs import Outcar
import logging
from concurrent.futures import ThreadPoolExecutor
from .vasp_interface import VaspInterface

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MigrationBarrierAnalyzer:
    """迁移势垒分析类"""
    
    def __init__(self,
                 vasp_interface: VaspInterface,
                 work_dir: str = "migration_analysis",
                 max_workers: int = 4):
        """
        初始化迁移势垒分析器
        
        Args:
            vasp_interface: VASP接口实例
            work_dir: 工作目录
            max_workers: 最大并行计算数
        """
        self.vasp = vasp_interface
        self.work_dir = os.path.abspath(work_dir)
        self.max_workers = max_workers
        
        # 创建工作目录
        os.makedirs(self.work_dir, exist_ok=True)
        
    def find_migration_paths(self,
                           structure: Structure,
                           migrating_specie: str,
                           max_paths: int = 10,
                           max_dist: float = 5.0) -> List[Dict]:
        """
        查找可能的迁移路径
        
        Args:
            structure: 结构
            migrating_specie: 迁移离子符号
            max_paths: 最大路径数
            max_dist: 最大迁移距离(Å)
            
        Returns:
            可能的迁移路径列表
        """
        # 使用FullPathMapper查找所有可能的迁移路径
        mapper = FullPathMapper(structure=structure,
                              migrating_specie=migrating_specie,
                              max_path_length=max_dist)
        
        paths = mapper.get_paths()
        
        # 按路径长度排序并限制数量
        paths = sorted(paths, key=lambda x: x["length"])[:max_paths]
        
        return [{
            "start_site": path["start_site"].as_dict(),
            "end_site": path["end_site"].as_dict(),
            "length": path["length"],
            "path_id": i
        } for i, path in enumerate(paths)]
    
    def calculate_migration_barrier(self,
                                  structure: Structure,
                                  start_site: Dict,
                                  end_site: Dict,
                                  n_images: int = 7,
                                  relax_endpoints: bool = True,
                                  path_id: Optional[str] = None) -> Dict:
        """
        计算特定迁移路径的势垒
        
        Args:
            structure: 初始结构
            start_site: 起始位置
            end_site: 终止位置
            n_images: NEB图像数量
            relax_endpoints: 是否优化端点
            path_id: 路径标识符
            
        Returns:
            势垒计算结果
        """
        try:
            # 创建计算目录
            if path_id is None:
                path_id = "path_" + str(hash(str(start_site) + str(end_site)))[:8]
            path_dir = os.path.join(self.work_dir, path_id)
            os.makedirs(path_dir, exist_ok=True)
            
            # 准备初始和终态结构
            initial_structure = structure.copy()
            final_structure = structure.copy()
            
            # 设置迁移离子位置
            initial_structure.replace(initial_structure.sites.index(start_site),
                                   start_site["species"][0]["element"],
                                   start_site["coords"])
            final_structure.replace(final_structure.sites.index(end_site),
                                  end_site["species"][0]["element"],
                                  end_site["coords"])
            
            # 如果需要，优化端点结构
            if relax_endpoints:
                logger.info("优化端点结构...")
                initial_dir = os.path.join(path_dir, "initial")
                final_dir = os.path.join(path_dir, "final")
                
                # 并行优化两个端点
                with ThreadPoolExecutor(max_workers=2) as executor:
                    initial_future = executor.submit(self._relax_structure,
                                                  initial_structure,
                                                  initial_dir)
                    final_future = executor.submit(self._relax_structure,
                                                final_structure,
                                                final_dir)
                    
                    initial_structure = initial_future.result()
                    final_structure = final_future.result()
            
            # 生成NEB路径
            logger.info("生成NEB路径...")
            neb = NEBPathfinder(start_struct=initial_structure,
                              end_struct=final_structure,
                              relax=True)
            
            images = neb.get_path(n_images)
            
            # 计算NEB
            logger.info("开始NEB计算...")
            neb_results = self.vasp.calculate_migration_barrier(
                initial_structure=initial_structure,
                final_structure=final_structure,
                n_images=n_images
            )
            
            # 分析结果
            if neb_results["is_converged"]:
                barrier = neb_results["barrier"]
                path_energies = neb_results["path"]
                
                # 计算迁移率
                from scipy.constants import k, h
                T = 300  # 室温
                k_B = 8.617333262145e-5  # 玻尔兹曼常数 (eV/K)
                rate = (k_B * T / h) * np.exp(-barrier / (k_B * T))
                
                results = {
                    "path_id": path_id,
                    "barrier": barrier,
                    "path_energies": path_energies,
                    "migration_rate": rate,
                    "path_length": np.linalg.norm(
                        np.array(end_site["coords"]) - np.array(start_site["coords"])
                    ),
                    "is_converged": True,
                    "images": [img.as_dict() for img in images]
                }
            else:
                results = {
                    "path_id": path_id,
                    "barrier": float('inf'),
                    "path_energies": [],
                    "migration_rate": 0.0,
                    "path_length": 0.0,
                    "is_converged": False,
                    "images": []
                }
            
            return results
            
        except Exception as e:
            logger.error(f"计算迁移势垒时出错: {e}")
            return {
                "path_id": path_id,
                "barrier": float('inf'),
                "path_energies": [],
                "migration_rate": 0.0,
                "path_length": 0.0,
                "is_converged": False,
                "images": [],
                "error": str(e)
            }
    
    def analyze_doping_effect(self,
                            base_structure: Structure,
                            dopant: str,
                            doping_sites: List[Dict],
                            migrating_specie: str,
                            concentrations: List[float]) -> Dict:
        """
        分析掺杂对迁移势垒的影响
        
        Args:
            base_structure: 基础结构
            dopant: 掺杂元素
            doping_sites: 可能的掺杂位置
            migrating_specie: 迁移离子
            concentrations: 掺杂浓度列表
            
        Returns:
            掺杂效应分析结果
        """
        results = {}
        
        for conc in concentrations:
            # 生成掺杂结构
            doped_structure = self._create_doped_structure(
                base_structure,
                dopant,
                doping_sites,
                concentration=conc
            )
            
            # 查找迁移路径
            paths = self.find_migration_paths(
                doped_structure,
                migrating_specie
            )
            
            # 计算每个路径的势垒
            path_results = []
            for path in paths:
                barrier = self.calculate_migration_barrier(
                    doped_structure,
                    path["start_site"],
                    path["end_site"],
                    path_id=f"doped_{conc}_{path['path_id']}"
                )
                path_results.append(barrier)
            
            # 统计分析
            valid_barriers = [r["barrier"] for r in path_results 
                            if r["is_converged"] and r["barrier"] != float('inf')]
            
            if valid_barriers:
                results[conc] = {
                    "min_barrier": min(valid_barriers),
                    "max_barrier": max(valid_barriers),
                    "avg_barrier": np.mean(valid_barriers),
                    "std_barrier": np.std(valid_barriers),
                    "path_results": path_results
                }
            else:
                results[conc] = {
                    "min_barrier": float('inf'),
                    "max_barrier": float('inf'),
                    "avg_barrier": float('inf'),
                    "std_barrier": 0.0,
                    "path_results": path_results
                }
        
        return results
    
    def analyze_vacancy_effect(self,
                             structure: Structure,
                             vacancy_sites: List[Dict],
                             migrating_specie: str) -> Dict:
        """
        分析空位对迁移势垒的影响
        
        Args:
            structure: 基础结构
            vacancy_sites: 可能的空位位置
            migrating_specie: 迁移离子
            
        Returns:
            空位效应分析结果
        """
        results = {}
        
        for i, vacancy in enumerate(vacancy_sites):
            # 创建含空位的结构
            vacancy_structure = structure.copy()
            vacancy_structure.remove_sites([vacancy["site_index"]])
            
            # 查找迁移路径
            paths = self.find_migration_paths(
                vacancy_structure,
                migrating_specie
            )
            
            # 计算每个路径的势垒
            path_results = []
            for path in paths:
                barrier = self.calculate_migration_barrier(
                    vacancy_structure,
                    path["start_site"],
                    path["end_site"],
                    path_id=f"vacancy_{i}_{path['path_id']}"
                )
                path_results.append(barrier)
            
            # 统计分析
            valid_barriers = [r["barrier"] for r in path_results 
                            if r["is_converged"] and r["barrier"] != float('inf')]
            
            if valid_barriers:
                results[f"vacancy_{i}"] = {
                    "min_barrier": min(valid_barriers),
                    "max_barrier": max(valid_barriers),
                    "avg_barrier": np.mean(valid_barriers),
                    "std_barrier": np.std(valid_barriers),
                    "vacancy_site": vacancy,
                    "path_results": path_results
                }
            else:
                results[f"vacancy_{i}"] = {
                    "min_barrier": float('inf'),
                    "max_barrier": float('inf'),
                    "avg_barrier": float('inf'),
                    "std_barrier": 0.0,
                    "vacancy_site": vacancy,
                    "path_results": path_results
                }
        
        return results
    
    def _relax_structure(self,
                        structure: Structure,
                        calc_dir: str) -> Structure:
        """优化结构"""
        self.vasp.prepare_input_files(
            structure,
            calc_type="relax",
            calc_dir=calc_dir
        )
        
        if self.vasp.run_calculation(calc_dir):
            results = self.vasp.get_results(calc_dir)
            return Structure.from_dict(results["final_structure"])
        else:
            return structure
    
    def _create_doped_structure(self,
                              structure: Structure,
                              dopant: str,
                              doping_sites: List[Dict],
                              concentration: float) -> Structure:
        """创建掺杂结构"""
        doped_structure = structure.copy()
        
        # 计算需要替换的原子数
        n_sites = len(doping_sites)
        n_dope = int(n_sites * concentration)
        
        # 随机选择掺杂位置
        doping_indices = np.random.choice(
            range(n_sites),
            size=n_dope,
            replace=False
        )
        
        # 执行掺杂
        for idx in doping_indices:
            site = doping_sites[idx]
            doped_structure.replace(
                site["site_index"],
                dopant,
                coords=site["coords"]
            )
        
        return doped_structure 