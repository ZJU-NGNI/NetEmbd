"""启发式算法求解器基类模块。

此模块定义了启发式算法求解器的基类接口，包括：
1. 优化器状态管理
2. 部署方案管理
3. 目标值计算
4. 通用工具方法

子类需要实现具体的solve()方法。

Typical usage example:

    class MyHeuristic(HeuristicOptimizer):
        def solve(self) -> Optional[BaseDeployment]:
            # 实现具体的启发式算法
            pass
    
    optimizer = MyHeuristic(network, task)
    deployment = optimizer.solve()
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np

from netembd.deployment import Deployment
from netembd.network import Network
from netembd.task import Task
from netembd.interfaces.base_optimizer import BaseOptimizer, OptimizerConfig, OptimizationStatus

class HeuristicOptimizer(BaseOptimizer):
    """启发式算法求解器基类，定义通用接口和状态管理"""
    
    def __init__(self, network: Network, task: Task, config: Optional[OptimizerConfig] = None):
        """初始化启发式算法求解器
        
        Args:
            network: 网络拓扑对象
            task: 任务DAG对象
            config: 优化器配置对象
        """
        self._network = network
        self._task = task
        self._config = config or OptimizerConfig()
        
        self._status = OptimizationStatus.NOT_STARTED
        self._best_deployment: Optional[Deployment] = None
        self._best_objective: Optional[float] = None
        self._interrupted = False
    
    @property
    def status(self) -> OptimizationStatus:
        """获取优化状态
        
        Returns:
            优化状态枚举值
        """
        return self._status
    
    @property
    def best_deployment(self) -> Optional[Deployment]:
        """获取最优部署方案
        
        Returns:
            部署方案对象，如果未找到可行解则返回None
        """
        return self._best_deployment
    
    @property
    def get_objective(self) -> Optional[float]:
        """获取最优目标值
        
        Returns:
            目标函数值，如果未找到可行解则返回None
        """
        return self._best_objective
    
    def solve(self) -> Optional[Deployment]:
        """求解VNF部署优化问题
        
        Returns:
            部署方案对象，如果未找到可行解则返回None
            
        Note:
            子类必须实现此方法，提供具体的启发式算法实现
        """
        raise NotImplementedError("子类必须实现solve方法")
    
    def _update_best_solution(self, deployment: Deployment) -> None:
        """更新最优解
        
        Args:
            deployment: 当前部署方案
        """
        if not deployment.is_feasible():
            return
            
        current_objective = deployment.calculate_total_latency()
        if self._best_objective is None or current_objective < self._best_objective:
            self._best_deployment = deployment
            self._best_objective = current_objective
    
    def interrupt(self) -> None:
        """中断求解过程"""
        self._interrupted = True
        # deployment.assign_vnf(vnf_id, old_node)
    
    def solve(self) -> Optional[Deployment]:
        """求解部署优化问题
        
        Returns:
            最优部署方案，如果未找到可行解则返回None
        """
        try:
            self._status = OptimizationStatus.RUNNING
            start_time = time.time()
            
            # 获取任务分层
            layers = self._get_task_layers()
            
            # 创建部署方案
            deployment = Deployment(self._network, self._task)
            assigned_vnfs = {}
            
            # 按层分配VNF
            for layer in layers:
                for vnf_id in layer:
                    # 计算节点得分
                    scores = self._calculate_node_scores(vnf_id, assigned_vnfs)
                    
                    # 选择得分最低的可行节点
                    best_node = min(scores.items(), key=lambda x: x[1])[0]
                    if scores[best_node] == float('inf'):
                        self._status = OptimizationStatus.INFEASIBLE
                        return None
                    
                    # 分配VNF
                    deployment.assign_vnf(vnf_id, best_node)
                    assigned_vnfs[vnf_id] = best_node
                    
                    # 检查时间限制
                    if time.time() - start_time > self._config.time_limit:
                        self._status = OptimizationStatus.TIME_LIMIT
                        break
                    
                    # 检查是否被中断
                    if self._interrupted:
                        self._status = OptimizationStatus.INTERRUPTED
                        break
            
            # 如果找到可行解，进行局部搜索优化
            if deployment.is_feasible():
                self._local_search(deployment)
                self._best_deployment = deployment
                self._best_objective = deployment.calculate_total_latency()
                self._status = OptimizationStatus.OPTIMAL
                return deployment
            else:
                self._status = OptimizationStatus.INFEASIBLE
                return None
        
        except Exception as e:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError(f"求解过程出错：{str(e)}")
    
    def interrupt(self) -> None:
        """中断求解过程"""
        self._interrupted = True