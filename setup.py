#!/usr/bin/env python3
"""
机器学习势能训练项目安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-potential-training",
    version="1.0.0",
    author="lunazhang",
    description="机器学习势能训练：钙钛矿晶界电导率提升研究",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-potential-training",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "mayavi>=4.8.0",
            "ovito>=3.8.0",
        ],
        "hpc": [
            "mpi4py>=3.1.0",
            "cupy>=10.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-train=ml_potential_training:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine learning, materials science, perovskite, conductivity, DFT, NequIP, MACE",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ml-potential-training/issues",
        "Documentation": "https://github.com/yourusername/ml-potential-training#readme",
        "Source": "https://github.com/yourusername/ml-potential-training",
    },
) 