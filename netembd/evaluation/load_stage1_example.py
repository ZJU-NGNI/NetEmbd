#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用保存的MonPlan阶段一结果的示例脚本

此脚本展示如何加载和使用monplan_stage1_evaluate.py生成的结果文件。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monplan_stage1_evaluate import load_stage1_result, list_available_results
from netembd.network.FatTree import FatTree
from netembd.network.Wan import Wan
from netembd.optimize.MonPlan import create_stage2_algorithm

def example_load_and_use_stage1_result():
    """示例：加载并使用阶段一结果"""
    print("=== MonPlan阶段一结果使用示例 ===")
    
    # 1. 查看可用的结果
    available_networks = list_available_results()
    print(f"\n可用的网络结果: {available_networks}")
    
    if not available_networks:
        print("没有找到可用的阶段一结果，请先运行 monplan_stage1_evaluate.py")
        return
    
    # 2. 选择一个网络进行演示
    network_name = available_networks[0]
    print(f"\n使用网络: {network_name}")
    
    try:
        # 3. 加载阶段一结果
        stage1_result = load_stage1_result(network_name)
        print(f"阶段一结果类型: {type(stage1_result)}")
        print(f"流覆盖率: {stage1_result.flow_coverage:.3f}")
        
        # 4. 重新创建对应的网络（用于阶段二测试）
        if network_name.startswith('FatTree_pod'):
            pod_num = int(network_name.split('pod')[1])
            network = FatTree(pod_num=pod_num)
            print(f"重新创建了FatTree网络 (pod_num={pod_num})")
        else:
            # WAN网络
            wan_dir = Path(__file__).parent.parent.parent / "input" / "topology" / "wan" / network_name
            network = Wan()
            network.load_nodes(str(wan_dir / "nodes.csv"))
            network.load_edges(str(wan_dir / "edges.csv"))
            network.update_programmable_and_control_nodes()
            print(f"重新创建了WAN网络 ({network_name})")
        
        # 5. 现在可以使用stage1_result进行阶段二测试
        print("\n阶段一结果已准备好用于阶段二测试")
        print("可以调用阶段二算法: create_stage2_algorithm('escala').solve(network, stage1_result, flows)")
        
    except Exception as e:
        print(f"加载结果失败: {e}")
        import traceback
        traceback.print_exc()

def example_batch_load_results():
    """示例：批量加载所有结果"""
    print("\n=== 批量加载结果示例 ===")
    
    available_networks = list_available_results()
    
    for network_name in available_networks:
        try:
            result = load_stage1_result(network_name)
            # print(f"{network_name}: 流覆盖率={result.flow_coverage:.3f}")
            print(result)
        except Exception as e:
            print(f"{network_name}: 加载失败 - {e}")

if __name__ == "__main__":
    example_load_and_use_stage1_result()
    example_batch_load_results()