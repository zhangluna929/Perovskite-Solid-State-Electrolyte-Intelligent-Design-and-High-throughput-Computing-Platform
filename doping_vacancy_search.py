import argparse
import itertools
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("请先安装 PyYAML : pip install pyyaml") from e

# 动态导入现有训练器
from ml_potential_training import MLPotentialTrainer


def build_logger(out_dir: Path):
    """为探索脚本创建简单日志器"""
    log_file = out_dir / "doping_vacancy_search.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("doping_search")


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 合并默认值
    defaults = {
        "data_dir": "data",
        "output_dir": "exploration_output",
        "doping_elements": ["Sr", "Ba", "Al"],
        "doping_concentrations": [0.01, 0.02, 0.03],
        "exploration_method": "grid",  # or random
        "random_trials": 10,
        "target_structures": 35,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


def enumerate_combinations(elements: List[str], concentrations: List[float], method: str, trials: int) -> List[Tuple[str, float]]:
    """根据探索策略生成(元素,浓度)组合列表"""
    combos = list(itertools.product(elements, concentrations))
    if method == "random":
        random.shuffle(combos)
        combos = combos[: trials]
    return combos


def main():
    parser = argparse.ArgumentParser(description="掺杂与空位自动探索脚本")
    parser.add_argument("--config", type=str, default="config/doping_search.yaml", help="YAML 配置文件路径")
    parser.add_argument("--method", type=str, choices=["grid", "random"], help="探索方法 (grid/random)")
    parser.add_argument("--trials", type=int, help="随机搜索的迭代次数")

    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    cfg = load_config(cfg_path)
    # 命令行优先级最高
    if args.method:
        cfg["exploration_method"] = args.method
    if args.trials:
        cfg["random_trials"] = args.trials

    base_output_dir = Path(cfg["output_dir"])
    base_output_dir.mkdir(exist_ok=True)

    logger = build_logger(base_output_dir)
    logger.info("开始掺杂/空位自动探索…")

    combos = enumerate_combinations(
        cfg["doping_elements"], cfg["doping_concentrations"], cfg["exploration_method"], cfg["random_trials"]
    )
    logger.info(f"本次将评估 {len(combos)} 组组合")

    summary = []

    for idx, (elem, conc) in enumerate(combos, 1):
        sub_out_dir = base_output_dir / f"{elem}_{conc:.3f}"
        logger.info(f"[{idx}/{len(combos)}] 处理组合: 元素={elem}, 浓度={conc:.3f}, 输出={sub_out_dir}")

        # 初始化训练器 (每次重新实例，保持简洁)
        trainer = MLPotentialTrainer(data_dir=cfg["data_dir"], output_dir=str(sub_out_dir))
        # 更新掺杂配置
        trainer.config["doping_elements"] = [elem]
        trainer.config["doping_concentrations"] = [conc]
        trainer.config["target_structures"] = cfg["target_structures"]

        # 1. 分析 Cif
        trainer.analyze_cif_files()
        selected_structures = trainer.select_representative_structures(cfg["target_structures"])

        # 2. 生成掺杂 + 空位结构
        doped = trainer.generate_doped_structures(selected_structures)
        vacancy = trainer.generate_vacancy_structures(doped)
        trainer.save_generated_structures()

        num_doped = sum(len(v) for v in doped.values())
        num_vac = sum(len(v) for v in vacancy.values())
        logger.info(f"  生成掺杂结构 {num_doped} 个，空位结构 {num_vac} 个")

        summary.append({
            "element": elem,
            "concentration": conc,
            "doped_structures": num_doped,
            "vacancy_structures": num_vac,
            "output_dir": str(sub_out_dir),
        })

    # 保存摘要
    summary_path = base_output_dir / "exploration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"探索完成，摘要已保存至 {summary_path}")


if __name__ == "__main__":
    main() 