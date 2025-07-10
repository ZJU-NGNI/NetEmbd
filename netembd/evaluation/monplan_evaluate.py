"""MonPlan阶段一和阶段二分离测试脚本

此脚本专门用于测试MonPlan框架的两个阶段：
1. 阶段一：测量点选择算法（MonPlan, SpeedPlan, MtpPlan）
2. 阶段二：数据收集路由算法（使用统一的MonPlan阶段一结果作为输入）

使用FatTree网络拓扑进行测试。

Usage:
    python test_stage1_stage2_separate.py
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from netembd.network.FatTree import FatTree
from netembd.optimize.MonPlan import (
    FlowGenerator, MonPlanEvaluator,
    create_stage1_algorithm, create_stage2_algorithm
)
from netembd.optimize.MonPlan.evaluation import Stage1Evaluator

def create_test_network(pod_num=4, edge_bandwidth=10000):
    """创建测试用的FatTree网络
    
    Returns:
        FatTree网络实例
    """
    print("正在创建FatTree网络...")
    
    # 创建4-pod FatTree网络
    network = FatTree(pod_num=pod_num,edge_bandwidth=edge_bandwidth)
    # network.visualize()
    print(f"网络创建完成: {len(network.get_nodes())}个节点, {len(network.get_edges())}条边")
    print(f"可编程节点数: {len([n for n in network.get_nodes() if network.get_node_resource(n).programmable])}")
    print(f"控制节点数: {len([n for n in network.get_nodes() if network.get_node_resource(n).control_node])}")
    
    return network

def generate_test_flows(network, num_flows=30):
    """生成测试流
    
    Args:
        network: 网络实例
        num_flows: 流数量
        
    Returns:
        生成的流列表
    """
    print(f"\n正在生成{num_flows}个测试流...")
    
    flow_generator = FlowGenerator(network, num_flows=num_flows)
    flows = flow_generator.generate_flows(num_flows=num_flows)
    
    print(f"流生成完成: {len(flows)}个流")
    return flows

def test_stage1_algorithms(network, flows):
    """测试阶段一算法
    
    Args:
        network: 网络实例
        flows: 测试流列表
        
    Returns:
        各算法的结果字典
    """
    print("\n=== 阶段一算法测试 ===")
    
    # 定义要测试的阶段一算法
    # stage1_algorithms = ["monplan", "speedplan", "mtpplan"]
    stage1_algorithms = ["monplan"]
    # 创建评估器
    evaluator = Stage1Evaluator()
    results = {}
    
    # 测试每个算法
    for alg_name in stage1_algorithms:
        print(f"\n--- 测试{alg_name}算法 ---")
        
        try:
            # 创建算法实例
            algorithm = create_stage1_algorithm(alg_name)
            
            # 执行算法
            start_time = time.time()
            result = algorithm.solve(network, flows)
            exec_time = time.time() - start_time
            
            # 评估结果
            metrics = evaluator.evaluate(result, flows, network, alg_name)
            results[alg_name] = {
                'result': result,
                'metrics': metrics,
                'execution_time': exec_time
            }
            
            # 打印结果
            print(f"执行时间: {exec_time:.3f}秒")
            print(f"流覆盖率: {metrics.flow_coverage_rate:.3f}")
            print(f"交换机利用率: {metrics.switch_utilization_rate:.3f}")
            print(f"平均控制跳数: {metrics.average_control_hops:.3f}")
            print(f"使用交换机数: {metrics.total_switches_used}")
            print(f"覆盖流数: {result.flow_coverage * len(flows)}/{len(flows)}")
            
        except Exception as e:
            print(f"算法{alg_name}执行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 比较算法性能
    if len(results) > 1:
        print("\n--- 阶段一算法性能对比 ---")
        
        # 提取评估指标进行比较
        metrics_only = {alg: data['metrics'] for alg, data in results.items()}
        comparison = evaluator.compare_algorithms(metrics_only)
        
        # 打印各指标的最佳算法
        print("\n各指标最佳算法:")
        for metric, best_alg in comparison["best_algorithm"].items():
            metric_name = {
                "flow_coverage_rate": "流覆盖率",
                "switch_utilization_rate": "交换机利用率",
                "average_control_hops": "平均控制跳数",
                "execution_time": "执行时间"
            }.get(metric, metric)
            print(f"  {metric_name}: {best_alg}")
        
        # 打印详细对比表格
        print("\n详细指标对比:")
        algorithms = list(results.keys())
        print(f"{'指标':<20} {' '.join(f'{alg:<15}' for alg in algorithms)}")
        print("-" * (20 + 15 * len(algorithms)))
        
        metric_names = {
            "flow_coverage_rate": "流覆盖率",
            "switch_utilization_rate": "交换机利用率",
            "average_control_hops": "平均控制跳数",
            "execution_time": "执行时间(s)"
        }
        
        for metric, display_name in metric_names.items():
            values = []
            for alg in algorithms:
                if metric == "execution_time":
                    val = results[alg]['execution_time']
                else:
                    val = getattr(results[alg]['metrics'], metric, 0)
                values.append(f"{val:.3f}")
            print(f"{display_name:<20} {' '.join(f'{val:<15}' for val in values)}")
    
    return results

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
            
            results[alg_name] = {
                'result': result,
                'execution_time': exec_time
            }
            
            # 打印结果
            print(f"执行时间: {exec_time:.3f}秒")
            print(f"总跳数: {result.total_hop_count}")
            print(f"拥塞率: {result.congestion_rate:.3f}")
            print(f"数据丢失率: {result.data_loss_rate:.3f}")
            print(f"数据路径数: {len(result.data_paths)}")
            
            # 打印测量点路径信息
            if hasattr(result, 'measurement_point_paths') and result.measurement_point_paths:
                print(f"测量点路径数: {len(result.measurement_point_paths)}")
                # 显示前几个路径作为示例
                for i, (mp, paths) in enumerate(list(result.measurement_point_paths.items())[:3]):
                    print(f"  测量点{mp}: {len(paths)}条路径")
                    for j, path in enumerate(paths[:2]):  # 只显示前2条路径
                        print(f"    路径{j+1}: {' -> '.join(map(str, path))}")
            
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
                if attr_name == "execution_time":
                    val = results[alg]['execution_time']
                elif attr_name == "data_paths_count":
                    val = len(results[alg]['result'].data_paths)
                else:
                    val = getattr(results[alg]['result'], attr_name, 0)
                
                if isinstance(val, float):
                    values.append(f"{val:.3f}")
                else:
                    values.append(f"{val}")
            
            print(f"{display_name:<20} {' '.join(f'{val:<15}' for val in values)}")
        
        # 找出最佳算法
        print("\n最佳算法:")
        
        # 最低拥塞率
        best_congestion = min(algorithms, key=lambda alg: results[alg]['result'].congestion_rate)
        print(f"  最低拥塞率: {best_congestion} ({results[best_congestion]['result'].congestion_rate:.3f})")
        
        # 最低数据丢失率
        best_loss = min(algorithms, key=lambda alg: results[alg]['result'].data_loss_rate)
        print(f"  最低数据丢失率: {best_loss} ({results[best_loss]['result'].data_loss_rate:.3f})")
        
        # 最快执行时间
        best_time = min(algorithms, key=lambda alg: results[alg]['execution_time'])
        print(f"  最快执行时间: {best_time} ({results[best_time]['execution_time']:.3f}s)")
    
    return results

def main():
    """主函数"""
    print("MonPlan阶段一和阶段二分离测试")
    print("=" * 50)
    
    try:
        # 1. 创建FatTree网络
        network = create_test_network(pod_num = 4)
        
        # 2. 生成测试流
        flows = generate_test_flows(network, num_flows=100000)
        # print(network.flow_generator.available_bandwidth)
        # network.visualize()
        # 3. 测试阶段一算法
        stage1_results = test_stage1_algorithms(network, flows)
        
        # 4. 获取MonPlan阶段一结果作为阶段二的统一输入
        if 'monplan' in stage1_results:
            monplan_stage1_result = stage1_results['monplan']['result']
            
            # 5. 测试阶段二算法
            stage2_results = test_stage2_algorithms(network, monplan_stage1_result, flows)
            
            print("\n=== 测试总结 ===")
            print(f"阶段一测试完成: {len(stage1_results)}个算法")
            print(f"阶段二测试完成: {len(stage2_results)}个算法")
            print("\n所有测试均使用相同的FatTree网络和流集合")
            print("阶段二测试均使用MonPlan阶段一的结果作为输入")
        else:
            print("\n错误: MonPlan阶段一算法执行失败，无法进行阶段二测试")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试执行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()