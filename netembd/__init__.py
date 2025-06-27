"""NetEmbd - 网络功能部署优化框架。

此包提供了一套完整的工具，用于将虚拟网络功能(VNF)高效地部署到物理网络节点上。
主要功能包括：

1. 网络拓扑管理：加载和管理物理网络的节点和边
2. 任务DAG管理：加载和管理VNF节点及其依赖关系
3. 部署优化：提供精确求解和启发式算法两种优化方法
4. 结果管理：保存和可视化优化结果

Typical usage example:

    from netembd.network import Network
    from netembd.task import Task
    from netembd.optimize import HeuristicOptimizer
    
    network = Network()
    network.load_nodes("nodes.csv")
    network.load_edges("edges.csv")
    
    task = Task()
    task.load_vnfs("vnfs.csv")
    task.load_dependencies("deps.csv")
    
    optimizer = HeuristicOptimizer(network, task)
    deployment = optimizer.solve()
"""

from .network.network import Network
from .task.task import Task
from .optimize.exact import ExactOptimizer
from .optimize.heuristic import HeuristicOptimizer
from .deployment.deployment import Deployment
from .interfaces.base_optimizer import OptimizerConfig

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # 主要组件
    "Network",
    "Task",
    "ExactOptimizer",
    "HeuristicOptimizer",
    "Deployment",
    "OptimizerConfig",
    
    # 版本信息
    "__version__",
    "__author__",
    "__email__"
]