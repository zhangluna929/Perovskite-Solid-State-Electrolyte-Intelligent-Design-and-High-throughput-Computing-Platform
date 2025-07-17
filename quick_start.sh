#!/bin/bash
# 机器学习势能训练项目快速启动脚本
# 用于钙钛矿材料晶界电导率提升研究

echo "🚀 机器学习势能训练项目快速启动"
echo "========================================"
echo ""

# 检查Python版本
echo "🐍 检查Python环境..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python版本: $python_version"

# 检查CUDA是否可用
echo ""
echo "🔧 检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA 可用:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  CUDA不可用，将使用CPU训练（训练时间会更长）"
fi

# 检查是否存在虚拟环境
echo ""
echo "🌟 环境设置..."
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "📦 安装项目依赖..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ 基础依赖安装完成"
else
    echo "⚠️  requirements.txt文件不存在，手动安装核心依赖..."
    pip install numpy pandas matplotlib pymatgen ase torch scikit-learn
fi

# 检查GPU支持
echo ""
echo "🔍 检查PyTorch GPU支持..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'GPU不可用')"

# 创建输出目录
echo ""
echo "📁 创建输出目录..."
mkdir -p ml_potential_output/{structures,dft_calculations,training,analysis}
echo "✅ 输出目录创建完成"

# 检查数据目录
echo ""
echo "📊 检查数据目录..."
if [ -d "data" ]; then
    cif_count=$(find data -name "*.cif" | wc -l)
    echo "找到 $cif_count 个CIF文件"
    if [ $cif_count -gt 0 ]; then
        echo "✅ 数据目录检查完成"
    else
        echo "⚠️  数据目录中没有找到CIF文件"
    fi
else
    echo "⚠️  数据目录不存在，请确保data目录包含CIF文件"
fi

# 运行项目
echo ""
echo "🎯 开始运行机器学习势能训练..."
echo "========================================"
echo ""

# 选择运行模式
echo "请选择运行模式："
echo "1. 完整流程（推荐）"
echo "2. 仅分析CIF文件"
echo "3. 仅生成结构"
echo "4. 仅设置DFT计算"
echo "5. 仅训练ML模型"
echo "6. 自定义运行"
echo ""
read -p "请输入选择 (1-6): " choice

case $choice in
    1)
        echo "🔄 运行完整流程..."
        python3 ml_potential_training.py
        ;;
    2)
        echo "🔍 仅分析CIF文件..."
        python3 -c "
from ml_potential_training import MLPotentialTrainer
trainer = MLPotentialTrainer()
trainer.analyze_cif_files()
print('CIF文件分析完成，结果保存在 ml_potential_output/analysis/')
"
        ;;
    3)
        echo "🏗️ 仅生成结构..."
        python3 -c "
from ml_potential_training import MLPotentialTrainer
trainer = MLPotentialTrainer()
trainer.analyze_cif_files()
structures = trainer.select_representative_structures(35)
trainer.generate_doped_structures(structures)
trainer.generate_vacancy_structures(trainer.doped_structures)
trainer.save_generated_structures()
print('结构生成完成，结果保存在 ml_potential_output/structures/')
"
        ;;
    4)
        echo "⚙️ 仅设置DFT计算..."
        python3 -c "
from ml_potential_training import MLPotentialTrainer
trainer = MLPotentialTrainer()
trainer.analyze_cif_files()
structures = trainer.select_representative_structures(35)
trainer.generate_doped_structures(structures)
trainer.generate_vacancy_structures(trainer.doped_structures)
dft_tasks = trainer.setup_dft_calculations()
print(f'DFT计算设置完成，共 {len(dft_tasks)} 个任务')
print('运行: cd ml_potential_output/dft_calculations && ./submit_all.sh')
"
        ;;
    5)
        echo "🤖 仅训练ML模型..."
        python3 -c "
from ml_potential_training import MLPotentialTrainer
trainer = MLPotentialTrainer()
config_file = trainer.setup_nequip_training()
print('NequIP训练设置完成')
print('运行: cd ml_potential_output/training && ./train_nequip.sh')
"
        ;;
    6)
        echo "🛠️ 自定义运行..."
        echo "请手动运行Python脚本或查看README.md了解详细用法"
        ;;
    *)
        echo "❌ 无效选择，运行完整流程..."
        python3 ml_potential_training.py
        ;;
esac

# 显示结果
echo ""
echo "🎉 训练流程完成！"
echo "========================================"
echo ""
echo "📊 输出结果："
echo "  - 结构分析: ml_potential_output/analysis/structure_summary.csv"
echo "  - DFT计算: ml_potential_output/dft_calculations/"
echo "  - ML训练: ml_potential_output/training/"
echo "  - 实验指导: ml_potential_output/analysis/project_report.md"
echo ""
echo "📋 下一步操作："
echo "  1. 查看分析结果: cat ml_potential_output/analysis/project_report.md"
echo "  2. 运行DFT计算: cd ml_potential_output/dft_calculations && ./submit_all.sh"
echo "  3. 训练ML模型: cd ml_potential_output/training && ./train_nequip.sh"
echo ""
echo "💡 获取帮助："
echo "  - 查看README: cat README.md"
echo "  - 技术支持: https://github.com/yourusername/ml-potential-training/issues"
echo ""
echo "🎯 预期成果："
echo "  - 晶界电导率提升: 30-50%"
echo "  - 训练误差目标: ≤5 meV atom⁻¹"
echo "  - 提供实验配方导航"
echo ""
echo "感谢使用机器学习势能训练项目！" 