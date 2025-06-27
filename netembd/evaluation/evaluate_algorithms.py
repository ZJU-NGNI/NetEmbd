import os
import time
from typing import Dict, List, Optional, Tuple

from netembd.network import Network
from netembd.task import Task
from netembd.optimize import OptimizerConfig
from netembd.optimize.speed.SpeedExact import SpeedExact
from netembd.optimize.speed.SpeedHeuristic import SpeedHeuristic
from netembd.optimize.hermes.HermesHeuristic import HermesHeuristic

def load_data(network_dir: str, task_dir: str) -> Tuple[Network, Task]:
    """加载网络拓扑和任务DAG数据
    
    Args:
        network_dir: 网络拓扑数据目录
        task_dir: 任务DAG数据目录
        
    Returns:
        网络拓扑对象和任务DAG对象
    """
    network = Network()
    network.load_nodes(os.path.join(network_dir, 'nodes.csv'))
    network.load_edges(os.path.join(network_dir, 'edges.csv'))
    
    task = Task()
    task.load_vnfs(os.path.join(task_dir, 'mats.csv'))
    task.load_dependencies(os.path.join(task_dir, 'deps.csv'))
    
    return network, task

def evaluate_algorithms(network_dir: str, task_dir: str) -> Dict[str, Dict[str, float]]:
    """评估不同算法的性能
    
    Args:
        network_dir: 网络拓扑数据目录
        task_dir: 任务DAG数据目录
        
    Returns:
        评估结果字典，key为算法名称，value为各项指标
    """
    # 加载数据
    network, task = load_data(network_dir, task_dir)
    
    # 配置优化器参数
    config = OptimizerConfig(
        time_limit=3600,  # 1小时时间限制
        gap_limit=0.01,   # 1%的gap限制
        verbose=False      # 不输出求解过程
    )
    
    # 初始化算法
    algorithms = {
        'SpeedExact': SpeedExact(network, task, config),
        'SpeedHeuristic': SpeedHeuristic(network, task, config),
        'HermesHeuristic': HermesHeuristic(network, task, epsilon1=100000, epsilon2=100, k=1, config=config)
    }
    
    results = {}
    
    # 评估每个算法
    for name, algorithm in algorithms.items():
        print(f'正在评估 {name}...')
        result = {}
        
        # 实验1：求解时间
        start_time = time.time()
        deployment = algorithm.solve()
        solve_time = time.time() - start_time
        result['solve_time'] = solve_time
        
        if deployment is None:
            print(f'{name} 未找到可行解')
            result.update({
                'stage_count': float('inf'),
                'total_latency': float('inf')
            })
        else:
            # deployment.visualize()
            # 实验3：计算部署方案中使用的stage数量
            assignments = deployment.get_assignments()
            # 使用(switch_id, stage_id)元组来唯一标识stage
            valid_stages = set()
            for assignment_info in assignments.values():
                if assignment_info.stage_id is not None:
                    valid_stages.add((assignment_info.node_id, assignment_info.stage_id))
            stage_count = len(valid_stages)
            result['stage_count'] = stage_count
            
            # 实验4：部署方案的latency
            total_latency = deployment.calculate_total_latency()
            result['total_latency'] = total_latency
            
        results[name] = result
        
    return results

def main():
    # 数据集路径
    datasets = [
        {
            'name': 'merge_6',
            'network': 'c:/Users/zedi2/OneDrive/code/baseline/src_v3/examples/FatTree_20',
            'task': 'c:/Users/zedi2/OneDrive/code/baseline/src_v3/examples/merge_6'
        }
    ]
    
    # 评估每个数据集
    for dataset in datasets:
        print(f'\n评估数据集: {dataset["name"]}')
        results = evaluate_algorithms(dataset['network'], dataset['task'])
        
        # 打印结果
        print('\n评估结果:')
        print('-' * 80)
        print(f'算法名称'.ljust(20), '求解时间(s)'.ljust(15), 'Stage数量'.ljust(15), '总延迟(ms)'.ljust(15))
        print('-' * 80)
        
        for name, result in results.items():
            print(
                f'{name}'.ljust(20),
                f'{result["solve_time"]:.2f}'.ljust(15),
                f'{result["stage_count"]:.0f}'.ljust(15),
                f'{result["total_latency"]:.2f}'.ljust(15)
            )
            
if __name__ == '__main__':
    main()