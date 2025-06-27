"""优化器接口定义。

此模块定义了优化器的核心抽象接口，包括：
1. 配置管理：优化参数设置
2. 求解控制：执行和中断优化过程
3. 状态跟踪：监控优化进度
4. 结果管理：获取最优解

Typical usage example:

    from netembd.interfaces import BaseOptimizer, OptimizerConfig
    
    class CustomOptimizer(BaseOptimizer):
        def solve(self) -> Optional[BaseDeployment]:
            # 自定义优化逻辑
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from netembd.interfaces.base_network import BaseNetwork
from netembd.interfaces.base_task import BaseTask
from netembd.interfaces.base_deployment import BaseDeployment

class OptimizationStatus(Enum):
    """优化状态枚举类"""
    NOT_STARTED = auto()  # 未开始
    RUNNING = auto()      # 运行中
    SOLVED = auto()       # 已找到可行解
    OPTIMAL = auto()      # 已找到最优解
    INFEASIBLE = auto()   # 问题无解
    ERROR = auto()        # 求解出错
    INTERRUPTED = auto()  # 已中断

@dataclass
class OptimizerConfig:
    """优化器配置数据类"""
    max_iterations: int = 100    # 最大迭代次数
    time_limit: int = 3600      # 时间限制（秒）
    random_seed: int = 42       # 随机种子
    verbose: bool = False       # 是否输出详细日志
    tolerance: float = 1e-6     # 收敛容差
    gap_limit: float = 0.01     # MIP Gap限制（用于整数规划）

class BaseOptimizer(ABC):
    """优化器抽象基类"""
    
    def __init__(
        self,
        network: BaseNetwork,
        task: BaseTask,
        config: Optional[OptimizerConfig] = None
    ):
        """初始化优化器
        
        Args:
            network: 网络拓扑对象
            task: 任务DAG对象
            config: 优化器配置
        """
        self.network = network
        self.task = task
        self.config = config or OptimizerConfig()
        self._status = OptimizationStatus.NOT_STARTED
        self._best_deployment = None
        self._best_objective = float('inf')
    
    @property
    def status(self) -> OptimizationStatus:
        """获取优化状态
        
        Returns:
            当前优化状态
        """
        return self._status
    
    @property
    def best_deployment(self) -> Optional[BaseDeployment]:
        """获取最优部署方案
        
        Returns:
            最优部署方案对象，如果未找到可行解则返回None
        """
        return self._best_deployment
    
    @property
    def best_objective(self) -> float:
        """获取最优目标值
        
        Returns:
            最优目标值（总通信延迟）
        """
        return self._best_objective
    
    @abstractmethod
    def solve(self) -> Optional[BaseDeployment]:
        """执行优化求解
        
        Returns:
            如果找到可行解则返回部署方案对象，否则返回None
        """
        pass
    
    @abstractmethod
    def interrupt(self) -> None:
        """中断优化过程"""
        pass
    
    # @abstractmethod
    # def objective_callback(self, deployment: BaseDeployment) -> None:
    #     """目标函数值回调
        
    #     在优化过程中，每找到一个更优解，就会调用此方法
        
    #     Args:
    #         deployment: 当前部署方案对象
    #     """
    #     pass

    @abstractmethod
    def get_objective(self, deployment: BaseDeployment) -> float:
        """目标函数定义
        
        子类必须实现此方法，定义具体的目标函数
        
        Args:
            deployment: 当前部署方案对象
        
        Returns:
            目标函数值
        """
        pass
