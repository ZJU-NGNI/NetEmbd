"""MonPlan阶段二批量测试脚本

此脚本用于批量测试MonPlan框架的阶段二算法，使用阶段一的MonPlan结果作为输入。
支持多种网络类型，并将结果保存到文件。

支持的网络类型：
1. FatTree网络：pod_num为16, 20, 24, 28, 32
2. WAN网络：input/topology/wan目录下的所有网络

输出文件：
- {network_name}_stage2_results.json: 完整的阶段二结果（JSON格式）
- all_networks_stage2_summary.json: 测试汇总信息

使用方法：
1. 批量测试所有网络：
   python monplan_stage2_evaluate.py
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from netembd.network.FatTree import FatTree
from netembd.network.Wan import Wan
from netembd.optimize.MonPlan import (
    FlowGenerator, MonPlanEvaluator,
    create_stage1_algorithm, create_stage2_algorithm
)
from monplan_stage1_evaluate import (
    create_fattree_network, create_wan_network,
    generate_test_flows, load_stage1_result
)

def test_stage2_algorithms(network, monplan_stage1_result, flows):
    """测试阶段二算法（使用统一的MonPlan阶段一结果）
    
    Args:
        network: 网络实例
        monplan_stage1_result: MonPlan阶段一的结果
        flows: 背景流量列表
        
    Returns:
        各算法的结果字典
    """
    print("\n=== 阶段二算法测试 ===")
    print("使用MonPlan阶段一结果作为统一输入")
    
    # 定义要测试的阶段二算法
    stage2_algorithms = ["escala", "random", "monplan", "intpath"]
    
    results = {}
    
    # 测试每个算法
    for alg_name in stage2_algorithms:
        print(f"\n--- 测试{alg_name}算法 ---")
        
        try:
            # 创建算法实例
            algorithm = create_stage2_algorithm(alg_name)
            
            # 执行算法
            start_time = time.time()
            result = algorithm.solve(network, monplan_stage1_result, flows)
            exec_time = time.time() - start_time
            
            # 准备结果数据
            results[alg_name] = {
                'total_hop_count': result.total_hop_count,
                'congestion_rate': result.congestion_rate,
                'data_loss_rate': result.data_loss_rate,
                'data_paths_count': len(result.data_paths),
                'execution_time': exec_time
            }
            
            # 如果有测量点路径信息，添加到结果中
            if hasattr(result, 'measurement_point_paths') and result.measurement_point_paths:
                results[alg_name]['measurement_points_count'] = len(result.measurement_point_paths)
            
            # 打印结果
            print(f"执行时间: {exec_time:.3f}秒")
            print(f"总跳数: {result.total_hop_count}")
            print(f"拥塞率: {result.congestion_rate:.3f}")
            print(f"数据丢失率: {result.data_loss_rate:.3f}")
            print(f"数据路径数: {len(result.data_paths)}")
            
        except Exception as e:
            print(f"算法{alg_name}执行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 比较算法性能
    if len(results) > 1:
        print("\n--- 阶段二算法性能对比 ---")
        
        algorithms = list(results.keys())
        print(f"{'指标':<20} {' '.join(f'{alg:<15}' for alg in algorithms)}")
        print("-" * (20 + 15 * len(algorithms)))
        
        # 比较各项指标
        metrics = [
            ("总跳数", "total_hop_count"),
            ("拥塞率", "congestion_rate"),
            ("数据丢失率", "data_loss_rate"),
            ("执行时间(s)", "execution_time"),
            ("数据路径数", "data_paths_count")
        ]
        
        for display_name, attr_name in metrics:
            values = []
            for alg in algorithms:
                val = results[alg][attr_name]
                if isinstance(val, float):
                    values.append(f"{val:.3f}")
                else:
                    values.append(f"{val}")
            print(f"{display_name:<20} {' '.join(f'{val:<15}' for val in values)}")
        
        # 找出最佳算法
        print("\n最佳算法:")
        
        # 最低拥塞率
        best_congestion = min(algorithms, key=lambda alg: results[alg]['congestion_rate'])
        print(f"  最低拥塞率: {best_congestion} ({results[best_congestion]['congestion_rate']:.3f})")
        
        # 最低数据丢失率
        best_loss = min(algorithms, key=lambda alg: results[alg]['data_loss_rate'])
        print(f"  最低数据丢失率: {best_loss} ({results[best_loss]['data_loss_rate']:.3f})")
        
        # 最快执行时间
        best_time = min(algorithms, key=lambda alg: results[alg]['execution_time'])
        print(f"  最快执行时间: {best_time} ({results[best_time]['execution_time']:.3f}s)")
    
    return results

def save_stage2_results(results, network_name, output_dir="stage2_results"):
    """保存阶段二结果到文件
    
    Args:
        results: 阶段二测试结果
        network_name: 网络名称
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 保存完整结果
    results_file = output_path / f"{network_name}_stage2_results.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"阶段二结果已保存到: {results_file}")
    return results_file

def test_single_network_stage2(network, network_name, stage1_result, flows):
    """测试单个网络的阶段二算法
    
    Args:
        network: 网络实例
        network_name: 网络名称
        stage1_result: 阶段一的MonPlan结果
        flows: 测试流列表
        
    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f"开始测试网络阶段二: {network_name}")
    print(f"{'='*60}")
    
    try:
        # 测试阶段二算法
        stage2_results = test_stage2_algorithms(network, stage1_result, flows)
        
        # 保存结果到文件
        save_stage2_results(stage2_results, network_name)
        
        print(f"\n网络 {network_name} 阶段二测试完成")
        return stage2_results
        
    except Exception as e:
        print(f"网络 {network_name} 阶段二测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_all_networks_stage2():
    """测试所有指定的网络的阶段二算法"""
    print("开始批量网络阶段二测试")
    print("="*80)
    
    all_results = {}
    
    # 测试FatTree网络
    fattree_pod_nums = [16, 20, 24, 28, 32]
    for pod_num in fattree_pod_nums:
        try:
            network, network_name = create_fattree_network(pod_num)
            flows = generate_test_flows(network, num_flows=10000)
            
            # 加载阶段一结果
            try:
                stage1_result = load_stage1_result(network_name)
                result = test_single_network_stage2(network, network_name, stage1_result, flows)
                if result:
                    all_results[network_name] = result
            except FileNotFoundError as e:
                print(f"跳过网络 {network_name}: {e}")
                continue
                
        except Exception as e:
            print(f"FatTree pod_num={pod_num} 测试失败: {e}")
            continue
    
    # 获取WAN网络列表
    wan_base_dir = Path(__file__).parent.parent.parent / "input" / "topology" / "wan"
    wan_networks = []
    
    if wan_base_dir.exists():
        for item in wan_base_dir.iterdir():
            if item.is_dir() and (item / "nodes.csv").exists() and (item / "edges.csv").exists():
                wan_networks.append(item.name)
    
    print(f"\n发现 {len(wan_networks)} 个WAN网络: {wan_networks}")
    
    # 测试WAN网络
    for wan_name in wan_networks:
        try:
            network, network_name = create_wan_network(wan_name)
            flows = generate_test_flows(network, num_flows=5000)  # WAN网络使用较少流数
            
            # 加载阶段一结果
            try:
                stage1_result = load_stage1_result(network_name)
                result = test_single_network_stage2(network, network_name, stage1_result, flows)
                if result:
                    all_results[network_name] = result
            except FileNotFoundError as e:
                print(f"跳过网络 {network_name}: {e}")
                continue
                
        except Exception as e:
            print(f"WAN网络 {wan_name} 测试失败: {e}")
            continue
    
    # 保存汇总结果
    summary_file = Path("stage2_results") / "all_networks_stage2_summary.json"
    summary_data = {
        'total_networks_tested': len(all_results),
        'fattree_networks': len([name for name in all_results.keys() if name.startswith('FatTree')]),
        'wan_networks': len([name for name in all_results.keys() if not name.startswith('FatTree')]),
        'networks_list': list(all_results.keys())
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"批量阶段二测试完成!")
    print(f"成功测试 {len(all_results)} 个网络")
    print(f"结果汇总保存到: {summary_file}")
    print(f"{'='*80}")
    
    return all_results

def main():
    """主函数"""
    test_all_networks_stage2()

if __name__ == "__main__":
    main()