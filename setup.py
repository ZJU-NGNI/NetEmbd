# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="netembd",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "gurobipy>=9.5.0",
        "matplotlib>=3.4.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",     # 单元测试
            "pytest-cov>=3.0.0",  # 测试覆盖率
            "black>=22.0.0",      # 代码格式化
            "pylint>=2.14.0"      # 代码质量检查
        ]
    },
    python_requires=">=3.8",
    author="Zedi Chen",
    author_email="chenzedi@zju.edu.cn",
    description="一个通用的网络功能部署优化框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)