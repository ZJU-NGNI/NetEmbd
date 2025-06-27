"""精确求解器基类模块。

此模块提供了基于整数规划的精确求解器基类，包括：
1. 使用Gurobi建立整数规划模型的基础框架
2. 提供决策变量和约束条件的通用接口
3. 管理优化状态和结果
4. 处理求解过程的异常情况

子类需要实现具体的优化逻辑，包括：
1. 自定义目标函数
2. 添加特定的约束条件
3. 优化资源分配策略

Typical usage example:

    class CustomExactOptimizer(ExactOptimizer):
        def solve(self):
            # 实现具体的优化逻辑
            pass
    
    optimizer = CustomExactOptimizer(network, task, config)
    deployment = optimizer.solve()
"""

from typing import Dict, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from netembd.deployment import Deployment
from netembd.network import Network
from netembd.task import Task
from netembd.interfaces.base_optimizer import BaseOptimizer, OptimizerConfig, OptimizationStatus

class ExactOptimizer(BaseOptimizer):
    """精确求解器基类，提供了使用Gurobi进行整数规划求解的基础框架"""
    
    def __init__(self, network: Network, task: Task, config: Optional[OptimizerConfig] = None):
        """初始化精确求解器基类
        
        Args:
            network: 网络拓扑对象
            task: 任务DAG对象
            config: 优化器配置对象，用于配置求解参数
        """
        self._network = network
        self._task = task
        self._config = config or OptimizerConfig()
        
        self._model: Optional[gp.Model] = None
        self._x: Optional[Dict[Tuple[int, int], gp.Var]] = None
        self._status = OptimizationStatus.NOT_STARTED
        self._best_deployment: Optional[Deployment] = None
        self._best_objective: Optional[float] = None
    
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
    
    def _create_model(self) -> None:
        """创建并初始化Gurobi模型"""
        self._model = gp.Model("netembd")
        self._model.setParam('TimeLimit', self._config.time_limit)
        self._model.setParam('MIPGap', self._config.gap_limit)
    
    def _create_variables(self) -> None:
        """创建决策变量"""
        self._x = {}
        for vnf_id in self._task.get_vnfs():
            for node_id in self._network.get_nodes():
                self._x[vnf_id, node_id] = self._model.addVar(
                    vtype=GRB.BINARY,
                    name=f'x_{vnf_id}_{node_id}'
                )
    
    def _add_assignment_constraints(self) -> None:
        """添加VNF分配约束"""
        for vnf_id in self._task.get_vnfs():
            self._model.addConstr(
                gp.quicksum(self._x[vnf_id, node_id]
                          for node_id in self._network.get_nodes()) == 1,
                name=f'assign_{vnf_id}'
            )
    
    def _add_resource_constraints(self) -> None:
        """添加资源约束"""
        for node_id in self._network.get_nodes():
            node_resource = self._network.get_node_resource(node_id)
            if node_resource:
                # ALU约束
                self._model.addConstr(
                    gp.quicksum(self._x[vnf_id, node_id] * self._task.get_vnf_resource(vnf_id).alu
                              for vnf_id in self._task.get_vnfs()) <= node_resource.alu,
                    name=f'alu_{node_id}'
                )
                # Stage约束
                self._model.addConstr(
                    gp.quicksum(self._x[vnf_id, node_id] * self._task.get_vnf_resource(vnf_id).stage
                              for vnf_id in self._task.get_vnfs()) <= node_resource.stage,
                    name=f'stage_{node_id}'
                )
                # SRAM约束
                self._model.addConstr(
                    gp.quicksum(self._x[vnf_id, node_id] * self._task.get_vnf_resource(vnf_id).sram
                              for vnf_id in self._task.get_vnfs()) <= node_resource.sram,
                    name=f'sram_{node_id}'
                )
    
    def _build_objective(self) -> None:
        """构建目标函数
        
        子类应该重写此方法以实现自定义的目标函数
        """
        obj = 0
        for source, target in self._task.get_dependencies():
            data_size = self._task.get_dependency_data_size(source, target)
            for source_node in self._network.get_nodes():
                for target_node in self._network.get_nodes():
                    path = self._network.get_shortest_path(source_node, target_node)
                    if path:
                        latency = self._network.get_path_latency(path)
                        obj += self._x[source, source_node] * self._x[target, target_node] * latency * data_size
        
        self._model.setObjective(obj, GRB.MINIMIZE)
    
    def _build_deployment(self) -> Optional[Deployment]:
        """根据求解结果构建部署方案"""
        deployment = Deployment(self._network, self._task)
        for vnf_id in self._task.get_vnfs():
            for node_id in self._network.get_nodes():
                if self._x[vnf_id, node_id].x > 0.5:
                    deployment.assign_vnf(vnf_id, node_id)
        
        return deployment if deployment.is_feasible() else None
    
    def solve(self) -> Optional[Deployment]:
        """求解部署优化问题
        
        Returns:
            最优部署方案，如果未找到可行解则返回None
            
        子类可以重写此方法以实现自定义的求解逻辑
        """
        try:
            # 构建和求解模型
            self._create_model()
            self._create_variables()
            self._add_assignment_constraints()
            self._add_resource_constraints()
            self._build_objective()
            
            # 开始求解
            self._status = OptimizationStatus.RUNNING
            self._model.optimize()
            
            # 处理求解结果
            if self._model.status == GRB.INFEASIBLE:
                self._status = OptimizationStatus.INFEASIBLE
                return None
            elif self._model.status == GRB.TIME_LIMIT:
                self._status = OptimizationStatus.TIME_LIMIT
            elif self._model.status != GRB.OPTIMAL:
                self._status = OptimizationStatus.ERROR
                return None
            
            # 构建部署方案
            deployment = self._build_deployment()
            if not deployment:
                self._status = OptimizationStatus.INFEASIBLE
                return None
            
            # 设置最优解状态
            self._status = OptimizationStatus.OPTIMAL
            self._best_deployment = deployment
            self._best_objective = self._model.objVal
            return deployment
        
        except Exception as e:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError(f"求解过程出错：{str(e)}")
    
    def interrupt(self) -> None:
        """中断求解过程"""
        if self._model and self._status == OptimizationStatus.RUNNING:
            self._model.terminate()
            self._status = OptimizationStatus.INTERRUPTED

if __name__ == '__main__':
    # 导入测试所需的模块
    import os
    import time
    from netembd.network import Network
    from netembd.task import Task
    from netembd.optimize.hermes import HermesHeuristic
    
    # 设置测试数据路径
    base_dir = r"C:\Users\zedi2\OneDrive\code\baseline\src_v3"
    # base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    examples_dir = os.path.join(base_dir, 'examples')
    
    # 加载网络拓扑
    network = Network()
    network.load_nodes(os.path.join(examples_dir, 'nodes.csv'))
    network.load_edges(os.path.join(examples_dir, 'edges.csv'))
    print('网络拓扑加载完成:')
    print(f'节点数: {len(network.get_nodes())}')
    print(f'边数: {len(network.get_edges())}')
    
    # 加载任务DAG
    task = Task()
    task.load_vnfs(os.path.join(examples_dir, 'mats.csv'))
    task.load_dependencies(os.path.join(examples_dir, 'deps.csv'))
    print('\n任务DAG加载完成:')
    print(f'VNF数: {len(task.get_vnfs())}')
    print(f'依赖数: {len(task.get_dependencies())}')
    
    # 创建优化器配置
    config = OptimizerConfig(
        max_iterations=1000,
        time_limit=60,
        random_seed=42,
        verbose=True,
        tolerance=1e-6
    )
    
    # 创建并运行优化器
    print('\n开始优化求解...')
    start_time = time.time()
    
    try:
        optimizer = ExactOptimizer(network, task, config=config)
        deployment = optimizer.solve()
    except Exception as e:
        print(f'\n求解失败: {str(e)}')
        deployment = None
    
    end_time = time.time()
    print(f'求解完成，耗时: {end_time - start_time:.2f}秒')
    print(f'优化状态: {optimizer.status}')
    
    if deployment and deployment.is_feasible():
        print('\n找到可行解:')
        print(f'总延迟: {deployment.calculate_total_latency():.2f}')
        print('\nVNF部署方案:')
        for vnf_id in task.get_vnfs():
            node_id = deployment.get_vnf_assignment(vnf_id).node_id
            print(f'VNF {vnf_id} -> Node {node_id}')
    else:
        print('\n未找到可行解')