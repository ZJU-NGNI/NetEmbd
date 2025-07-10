"""MonPlan阶段一批量测试脚本

此脚本用于批量测试MonPlan框架的阶段一算法（MonPlan, SpeedPlan, MtpPlan），
支持多种网络类型，并将结果保存到文件供阶段二使用。

支持的网络类型：
1. FatTree网络：pod_num为16, 20, 24, 28, 32
2. WAN网络：input/topology/wan目录下的所有网络

输出文件：
- {network_name}_stage1_full_results.json: 完整的阶段一结果（JSON格式）
- {network_name}_monplan_stage1_result.pkl: MonPlan阶段一结果（pickle格式，供stage2使用）
- all_networks_summary.json: 测试汇总信息

使用方法：
1. 批量测试所有网络：
   python monplan_stage1_evaluate.py
   
2. 在其他脚本中加载结果：
   from monplan_stage1_evaluate import load_stage1_result
   result = load_stage1_result('FatTree_pod16')
   
3. 查看可用结果：
   from monplan_stage1_evaluate import list_available_results
   networks = list_available_results()
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path

from netembd.optimize.MonPlan.MonPlan import Stage1Algorithm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from netembd.network.FatTree import FatTree
from netembd.network.Wan import Wan
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

def create_fattree_network(pod_num, edge_bandwidth=10000):
    """创建FatTree网络
    
    Args:
        pod_num: Pod数量
        edge_bandwidth: 边带宽
        
    Returns:
        FatTree网络实例和网络名称
    """
    print(f"正在创建FatTree网络 (pod_num={pod_num})...")
    network = FatTree(pod_num=pod_num, edge_bandwidth=edge_bandwidth)
    network_name = f"FatTree_pod{pod_num}"
    print(f"网络创建完成: {len(network.get_nodes())}个节点, {len(network.get_edges())}条边")
    return network, network_name

def create_wan_network(wan_name):
    """创建WAN网络
    
    Args:
        wan_name: WAN网络名称
        
    Returns:
        WAN网络实例和网络名称
    """
    print(f"正在创建WAN网络 ({wan_name})...")
    
    # 构建文件路径
    wan_dir = Path(__file__).parent.parent.parent / "input" / "topology" / "wan" / wan_name
    nodes_file = wan_dir / "nodes.csv"
    edges_file = wan_dir / "edges.csv"
    
    if not nodes_file.exists() or not edges_file.exists():
        raise FileNotFoundError(f"WAN网络文件不存在: {wan_dir}")
    
    # 创建WAN网络并加载数据
    network = Wan()
    network.load_nodes(str(nodes_file))
    network.load_edges(str(edges_file))
    network.update_programmable_and_control_nodes()
    
    print(f"网络创建完成: {len(network.get_nodes())}个节点, {len(network.get_edges())}条边")
    print(f"可编程节点数: {len([n for n in network.get_nodes() if network.get_node_resource(n).programmable])}")
    print(f"控制节点数: {len([n for n in network.get_nodes() if network.get_node_resource(n).control_node])}")
    
    return network, wan_name

def generate_test_flows(network, num_flows=3000000):
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

def save_stage1_results(results, network_name, output_dir="stage1_results"):
    """保存阶段一结果到文件
    
    Args:
        results: 阶段一测试结果
        network_name: 网络名称
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 保存完整结果（包含所有算法）
    full_results_file = output_path / f"{network_name}_stage1_full_results.json"
    
    # 准备可序列化的结果数据
    serializable_results = {}
    for alg_name, data in results.items():
        serializable_results[alg_name] = {
            'metrics': {
                'flow_coverage_rate': data['metrics'].flow_coverage_rate,
                'switch_utilization_rate': data['metrics'].switch_utilization_rate,
                'average_control_hops': data['metrics'].average_control_hops,
                'total_switches_used': data['metrics'].total_switches_used
            },
            'execution_time': data['execution_time']
        }
    
    # 保存JSON格式的结果
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # 如果有MonPlan结果，单独保存用于stage2
    if 'monplan' in results:
        monplan_result = results['monplan']['result']
        
        # 保存pickle格式（用于stage2）
        monplan_pickle_file = output_path / f"{network_name}_monplan_stage1_result.pkl"
        with open(monplan_pickle_file, 'wb') as f:
            pickle.dump(monplan_result, f)
        
        print(f"MonPlan阶段一结果已保存到: {monplan_pickle_file}")
    
    print(f"完整阶段一结果已保存到: {full_results_file}")
    return full_results_file

def test_single_network(network, network_name, num_flows=10000):
    """测试单个网络
    
    Args:
        network: 网络实例
        network_name: 网络名称
        num_flows: 流数量
        
    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f"开始测试网络: {network_name}")
    print(f"{'='*60}")
    
    try:
        # 生成测试流
        flows = generate_test_flows(network, num_flows=num_flows)
        
        # 测试阶段一算法
        stage1_results = test_stage1_algorithms(network, flows)
        
        # 保存结果到文件
        save_stage1_results(stage1_results, network_name)
        
        print(f"\n网络 {network_name} 测试完成")
        return stage1_results
        
    except Exception as e:
        print(f"网络 {network_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_all_networks():
    """测试所有指定的网络"""
    print("开始批量网络测试")
    print("="*80)
    
    all_results = {}
    
    # 测试FatTree网络
    fattree_pod_nums = [16, 20, 24, 28, 32]
    for pod_num in fattree_pod_nums:
        try:
            network, network_name = create_fattree_network(pod_num)
            result = test_single_network(network, network_name, num_flows=10000)
            if result:
                all_results[network_name] = result
        except Exception as e:
            print(f"FatTree pod_num={pod_num} 创建失败: {e}")
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
            result = test_single_network(network, network_name, num_flows=5000)  # WAN网络使用较少流数
            if result:
                all_results[network_name] = result
        except Exception as e:
            print(f"WAN网络 {wan_name} 创建失败: {e}")
            continue
    
    # 保存汇总结果
    summary_file = Path("stage1_results") / "all_networks_summary.json"
    summary_data = {
        'total_networks_tested': len(all_results),
        'fattree_networks': len([name for name in all_results.keys() if name.startswith('FatTree')]),
        'wan_networks': len([name for name in all_results.keys() if not name.startswith('FatTree')]),
        'networks_list': list(all_results.keys())
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"批量测试完成!")
    print(f"成功测试 {len(all_results)} 个网络")
    print(f"结果汇总保存到: {summary_file}")
    print(f"{'='*80}")
    
    return all_results

def load_stage1_result(network_name, results_dir="stage1_results"):
    """加载指定网络的MonPlan阶段一结果
    
    Args:
        network_name: 网络名称
        results_dir: 结果目录
        
    Returns:
        MonPlan阶段一结果对象
    """
    results_path = Path(results_dir)
    pickle_file = results_path / f"{network_name}_monplan_stage1_result.pkl"
    
    if not pickle_file.exists():
        raise FileNotFoundError(f"找不到网络 {network_name} 的MonPlan阶段一结果文件: {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        result = pickle.load(f)
    
    print(f"已加载网络 {network_name} 的MonPlan阶段一结果")
    return result

def list_available_results(results_dir="stage1_results"):
    """列出可用的阶段一结果
    
    Args:
        results_dir: 结果目录
        
    Returns:
        可用网络名称列表
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    
    available_networks = []
    for file in results_path.glob("*_monplan_stage1_result.pkl"):
        network_name = file.stem.replace("_monplan_stage1_result", "")
        available_networks.append(network_name)
    
    return available_networks

def main():
    """主函数"""
    print("MonPlan阶段一和阶段二分离测试")
    print("=" * 50)
    
    try:
        # 1. 创建FatTree网络
        network = create_test_network()
        
        # 2. 生成测试流
        flows = generate_test_flows(network, num_flows=100)
        
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
    # 运行批量测试
    # test_all_networks()
    network, network_name = create_fattree_network(pod_num=6)
    print(network_name)
    flows = generate_test_flows(network, num_flows=100)
    stage1_results = test_stage1_algorithms(network, flows)
    save_stage1_results(stage1_results, network_name)

