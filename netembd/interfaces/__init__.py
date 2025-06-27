"""核心接口定义模块。

此模块定义了系统的核心抽象接口，包括：
1. BaseNetwork：网络拓扑的抽象基类
2. BaseTask：任务DAG的抽象基类
3. BaseDeployment：部署方案的抽象基类
4. BaseOptimizer：优化器的抽象基类

这些接口确保了系统各组件之间的解耦和可扩展性。每个接口都定义了：
- 必需的属性和方法
- 类型注解
- 详细的文档字符串

Typical usage example:

    from netembd.interfaces import BaseNetwork, BaseTask
    
    class CustomNetwork(BaseNetwork):
        def load_nodes(self, node_csv: str) -> None:
            # 自定义节点加载逻辑
            pass
    
    class CustomTask(BaseTask):
        def load_vnfs(self, vnf_csv: str) -> None:
            # 自定义VNF加载逻辑
            pass
"""

from .base_network import BaseNetwork, NodeResource
from .base_task import BaseTask, VNFResource, VNFDependency
from .base_deployment import BaseDeployment, Assignment
from .base_optimizer import (
    BaseOptimizer,
    OptimizerConfig,
    OptimizationStatus
)

__all__ = [
    # 网络拓扑接口
    "BaseNetwork",
    "NodeResource",
    
    # 任务DAG接口
    "BaseTask",
    "VNFResource",
    "VNFDependency",
    
    # 部署方案接口
    "BaseDeployment",
    "Assignment",
    
    # 优化器接口
    "BaseOptimizer",
    "OptimizerConfig",
    "OptimizationStatus"
]