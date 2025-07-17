"""
钙钛矿材料优化主程序
用于分析和优化钙钛矿材料的掺杂和空位结构
"""

import os
import json
from typing import Dict, List
from doping_analysis import DopingAnalyzer
from vacancy_analysis import VacancyAnalyzer

def analyze_base_structure(structure_file: str) -> Dict:
    """分析基础结构"""
    print(f"正在分析基础结构: {structure_file}")
    
    # 创建掺杂分析器
    doping_analyzer = DopingAnalyzer(
        structure_file=structure_file,
        doping_elements=["Sr", "Ba", "Al"],
        concentration_range=[0.01, 0.03]
    )
    
    # 分析结构特征
    structure_info = doping_analyzer.analyze_structure()
    print("\n结构特征:")
    print(f"- 晶格参数: {structure_info['lattice_params']}")
    print(f"- A位数量: {structure_info['a_sites']}")
    print(f"- B位数量: {structure_info['b_sites']}")
    
    return structure_info

def optimize_doping(structure_file: str) -> Dict:
    """优化掺杂配置"""
    print("\n开始掺杂优化...")
    
    # 创建掺杂分析器
    doping_analyzer = DopingAnalyzer(
        structure_file=structure_file,
        doping_elements=["Sr", "Ba", "Al"],
        concentration_range=[0.01, 0.03]
    )
    
    # 生成掺杂结构
    doped_structures = doping_analyzer.generate_doped_structures()
    print(f"生成了 {len(doped_structures)} 个掺杂结构")
    
    # 评估性能
    properties = doping_analyzer.evaluate_properties(doped_structures)
    
    # 获取优化建议
    suggestions = doping_analyzer.get_optimization_suggestions()
    print("\n掺杂优化建议:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n方案 {i}:")
        print(f"- 掺杂元素: {suggestion['doping_element']}")
        print(f"- 浓度: {suggestion['concentration']:.1%}")
        print(f"- 预期提升: {suggestion['expected_enhancement']:.1%}")
        
    return {"doped_structures": len(doped_structures),
            "suggestions": suggestions}

def analyze_vacancies(structure_file: str) -> Dict:
    """分析空位结构"""
    print("\n开始空位分析...")
    
    # 创建空位分析器
    vacancy_analyzer = VacancyAnalyzer(
        structure_file=structure_file,
        vacancy_types=["O", "Li", "La"]
    )
    
    # 分析配位环境
    coordination = vacancy_analyzer.analyze_coordination()
    print("\n配位环境分析完成")
    
    # 生成空位结构
    vacancy_structures = vacancy_analyzer.generate_vacancy_structures()
    print(f"生成了 {len(vacancy_structures)} 个空位结构")
    
    # 分析稳定性
    stability = vacancy_analyzer.analyze_stability()
    print("\n空位稳定性分析:")
    for name, score in stability.items():
        print(f"- {name}: {score:.3f}")
    
    # 评估电导率
    conductivity = vacancy_analyzer.evaluate_conductivity()
    print("\n电导率评估:")
    for name, value in conductivity.items():
        print(f"- {name}: {value:.2e} S/cm")
        
    return {"vacancy_structures": len(vacancy_structures),
            "stability": stability,
            "conductivity": conductivity}

def save_results(results: Dict, output_dir: str):
    """保存分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON文件
    output_file = os.path.join(output_dir, "optimization_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    """主函数"""
    # 设置输入输出路径
    base_structure = "data/LiLaTiO4.cif"
    output_dir = "optimization_results"
    
    # 1. 分析基础结构
    structure_info = analyze_base_structure(base_structure)
    
    # 2. 优化掺杂
    doping_results = optimize_doping(base_structure)
    
    # 3. 分析空位
    vacancy_results = analyze_vacancies(base_structure)
    
    # 4. 保存结果
    results = {
        "structure_info": structure_info,
        "doping_results": doping_results,
        "vacancy_results": vacancy_results
    }
    save_results(results, output_dir)

if __name__ == "__main__":
    main() 