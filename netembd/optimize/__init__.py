"""部署优化算法模块。

此模块提供了两种部署优化算法：
1. 精确求解：基于整数规划的精确优化方法
2. 启发式算法：基于任务分层和局部搜索的快速优化方法

两种算法都支持：
- 考虑节点资源约束
- 考虑通信带宽和延迟
- 支持自定义优化参数
- 提供求解状态和结果查询

Typical usage example:

    # 使用精确求解
    from netembd.optimize import ExactOptimizer, OptimizerConfig
    optimizer = ExactOptimizer(network, task, OptimizerConfig(time_limit=3600))
    deployment = optimizer.solve()
    
    # 使用启发式算法
    from netembd.optimize import HeuristicOptimizer
    optimizer = HeuristicOptimizer(network, task)
    deployment = optimizer.solve()
"""

from .exact import ExactOptimizer
from .heuristic import HeuristicOptimizer
from ..interfaces.base_optimizer import OptimizerConfig

__all__ = [
    "ExactOptimizer",
    "HeuristicOptimizer",
    "OptimizerConfig"
]