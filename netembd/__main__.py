"""NetEmbd项目的主入口模块。

此模块提供命令行接口，用于：
1. 加载网络拓扑和任务DAG
2. 选择优化算法（精确求解或启发式）
3. 求解部署优化问题
4. 保存和可视化结果

Typical usage example:

    python -m netembd optimize \
        --network-nodes nodes.csv \
        --network-edges edges.csv \
        --task-vnfs vnfs.csv \
        --task-deps deps.csv \
        --algorithm speed_exact \
        --output solution.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from netembd.network.network import Network
from netembd.task.task import Task
from netembd.optimize.speed.SpeedExact import SpeedExact
from netembd.optimize.speed.SpeedHeuristic import SpeedHeuristic
from netembd.optimize.hermes.HermesHeuristic import HermesHeuristic
from netembd.deployment.deployment import Deployment
from netembd.interfaces.base_optimizer import OptimizerConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize(
    network_nodes: str,
    network_edges: str,
    task_vnfs: str,
    task_deps: str,
    algorithm: str = 'speed_heuristic',
    output: Optional[str] = None,
    time_limit: int = 3600,
    gap_limit: float = 0.01,
    verbose: bool = False
) -> Optional[Deployment]:
    """执行部署优化
    
    Args:
        network_nodes: 网络节点配置文件路径
        network_edges: 网络边配置文件路径
        task_vnfs: 任务VNF配置文件路径
        task_deps: 任务依赖配置文件路径
        algorithm: 优化算法选择（'speed_exact'、'speed_heuristic'或'hermes_heuristic'）
        output: 输出文件路径
        time_limit: 求解时间限制（秒）
        gap_limit: 求解gap限制
        verbose: 是否输出详细日志
        
    Returns:
        如果找到可行解，返回对应的部署方案；否则返回None
    """
    try:
        # 加载网络拓扑
        logger.info("加载网络拓扑...")
        network = Network()
        network.load_nodes(network_nodes)
        network.load_edges(network_edges)
        logger.info(
            f"网络拓扑加载完成：{len(network.get_nodes())}个节点，"
            f"{len(network.get_edges())}条边"
        )
        
        # 加载任务DAG
        logger.info("加载任务DAG...")
        task = Task()
        task.load_vnfs(task_vnfs)
        task.load_dependencies(task_deps)
        logger.info(
            f"任务DAG加载完成：{len(task.get_vnfs())}个VNF，"
            f"{len(task.get_dependencies())}个依赖"
        )
        
        # 选择并配置优化算法
        logger.info(f"使用{algorithm}算法求解...")
        config = OptimizerConfig(
            time_limit=time_limit,
            gap_limit=gap_limit,
            verbose=verbose
        )
        
        if algorithm == 'speed_exact':
            optimizer = SpeedExact(network, task, config)
        elif algorithm == 'speed_heuristic':
            optimizer = SpeedHeuristic(network, task, config)
        else:  # hermes_heuristic
            optimizer = HermesHeuristic(network, task, epsilon1=100000, epsilon2=100, k=1, config=config)
        
        # 求解优化问题
        start_time = time.time()
        deployment = optimizer.solve()
        solve_time = time.time() - start_time
        
        if deployment:
            logger.info(
                f"找到可行解：总通信延迟={deployment.calculate_total_latency():.2f}ms，"
                f"求解时间={solve_time:.2f}s"
            )
            
            # 保存结果
            if output:
                deployment.save_to_csv(output)
                logger.info(f"部署方案已保存到{output}")
            
            return deployment
        else:
            logger.error("未找到可行解")
            return None
        
    except Exception as e:
        logger.error(f"优化失败：{str(e)}")
        return None

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="NetEmbd网络功能部署优化工具"
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(
        dest='command',
        help='可用命令'
    )
    
    # optimize命令
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='执行部署优化'
    )
    optimize_parser.add_argument(
        '--network-nodes',
        required=True,
        help='网络节点配置文件路径'
    )
    optimize_parser.add_argument(
        '--network-edges',
        required=True,
        help='网络边配置文件路径'
    )
    optimize_parser.add_argument(
        '--task-vnfs',
        required=True,
        help='任务VNF配置文件路径'
    )
    optimize_parser.add_argument(
        '--task-deps',
        required=True,
        help='任务依赖配置文件路径'
    )
    optimize_parser.add_argument(
        '--algorithm',
        choices=['speed_exact', 'speed_heuristic', 'hermes_heuristic'],
        default='speed_heuristic',
        help='优化算法选择'
    )
    optimize_parser.add_argument(
        '--output',
        help='输出文件路径'
    )
    optimize_parser.add_argument(
        '--time-limit',
        type=int,
        default=3600,
        help='求解时间限制（秒）'
    )
    optimize_parser.add_argument(
        '--gap-limit',
        type=float,
        default=0.01,
        help='求解gap限制'
    )
    optimize_parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细日志'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    if args.command == 'optimize':
        optimize(
            network_nodes=args.network_nodes,
            network_edges=args.network_edges,
            task_vnfs=args.task_vnfs,
            task_deps=args.task_deps,
            algorithm=args.algorithm,
            output=args.output,
            time_limit=args.time_limit,
            gap_limit=args.gap_limit,
            verbose=args.verbose
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    import sys
    import os
    # 将项目根目录添加到Python路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, project_root)
    main()