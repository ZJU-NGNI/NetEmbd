"""MonPlan网络测量优化框架主模块。

此模块实现了MonPlan网络测量优化框架，采用组合模式设计，包括：
1. 两阶段优化：测量点选择和数据收集
2. Flow初始化：支持DCN和WAN网络的Flow生成
3. 评估指标：独立的评估模块
4. 多种算法对比：每个阶段支持多种算法实现

Typical usage example:

    from netembd.optimize.MonPlan import MonPlan
    from netembd.optimize.MonPlan.measurement_point_selection import MonPlanStage1
from netembd.optimize.MonPlan.data_collection_routing import EscalaStage2
    
    monplan = MonPlan(network)
    monplan.set_stage1_algorithm(MonPlanStage1())
    monplan.set_stage2_algorithm(EscalaStage2())
    result = monplan.optimize()
"""

import random
from tabnanny import verbose
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from netembd.network import Network, FlowGenerator, Flow
from netembd.network.FatTree import FatTree
from netembd.network.Wan import Wan

@dataclass
class MeasurementPoint:
    """测量点数据类"""
    node_id: int
    measurement_types: List[str]  # ['sketch', 'INT'] 或其子集
    covered_flows: List[int]  # 覆盖的流ID列表
    control_node_distance: int  # 到最近控制节点的跳数

@dataclass
class Stage1Result:
    """阶段一结果数据类"""
    measurement_points: List[MeasurementPoint]
    flow_coverage: float  # 流覆盖率
    total_switches_used: int  # 使用的交换机数量
    total_control_hops: int  # 总控制跳数
    execution_time: float  # 执行时间
    objective_value: float  # 目标函数值

@dataclass
class Stage2Result:
    """阶段二结果数据类"""
    data_paths: Dict[int, List[List[int]]]  # 测量点到控制节点的数据路径
    total_hop_count: int  # 总跳数
    congestion_rate: float  # 拥塞率
    data_loss_rate: float  # 数据丢失率
    execution_time: float  # 执行时间

@dataclass
class MonPlanResult:
    """MonPlan完整结果数据类"""
    flows: List[Flow]  # 生成的流列表
    stage1_result: Stage1Result  # 阶段一结果
    stage2_result: Stage2Result  # 阶段二结果
    total_execution_time: float  # 总执行时间

class Stage1Algorithm(ABC):
    """阶段一算法抽象基类"""
    
    @abstractmethod
    def solve(self, network: Network, flows: List[Flow]) -> Stage1Result:
        """求解测量点选择问题
        
        Args:
            network: 网络拓扑
            flows: 网络流列表
            
        Returns:
            阶段一求解结果
        """
        pass

class Stage2Algorithm(ABC):
    """阶段二算法抽象基类"""
    
    @abstractmethod
    def solve(self, network: Network, stage1_result: Stage1Result, flows: List[Flow] = None) -> Stage2Result:
        """求解数据收集路径问题
        
        Args:
            network: 网络拓扑
            stage1_result: 阶段一的结果
            flows: 背景流量列表，用于拥塞率计算
            
        Returns:
            阶段二求解结果
        """
        pass


class MonPlan:
    """MonPlan主框架类，采用组合模式"""
    
    def __init__(self, flow_generator: FlowGenerator, 
                 stage1_algorithm: Optional[Stage1Algorithm] = None,
                 stage2_algorithm: Optional[Stage2Algorithm] = None):
        """初始化MonPlan框架
        
        Args:
            flow_generator: 流生成器
            stage1_algorithm: 阶段一算法，可选
            stage2_algorithm: 阶段二算法，可选
        """
        self.network = flow_generator.network
        self.flow_generator = flow_generator
        self.stage1_algorithm = stage1_algorithm
        self.stage2_algorithm = stage2_algorithm
        
    def set_stage1_algorithm(self, algorithm: Stage1Algorithm) -> None:
        """设置阶段一算法
        
        Args:
            algorithm: 阶段一算法实例
        """
        self.stage1_algorithm = algorithm
    
    def set_stage2_algorithm(self, algorithm: Stage2Algorithm) -> None:
        """设置阶段二算法
        
        Args:
            algorithm: 阶段二算法实例
        """
        self.stage2_algorithm = algorithm
    
    def set_flow_generator(self, generator: FlowGenerator) -> None:
        """设置流生成器
        
        Args:
            generator: 流生成器实例
        """
        self.flow_generator = generator
    
    def optimize(self, flows: Optional[List[Flow]] = None, num_flows: int = 100) -> MonPlanResult:
        """执行完整的两阶段优化
        
        Args:
            flows: 网络流列表，如果为None则自动生成
            num_flows: 生成的流数量，当flows为None时使用
            
        Returns:
            MonPlan完整结果
            
        Raises:
            ValueError: 算法未设置
        """
        if self.stage1_algorithm is None:
            raise ValueError("阶段一算法未设置")
        if self.stage2_algorithm is None:
            raise ValueError("阶段二算法未设置")
        
        # 生成或使用提供的流
        if flows is None:
            flows = self.flow_generator.generate_flows(num_flows=num_flows)
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行阶段一
        stage1_result = self.stage1_algorithm.solve(self.network, flows)
        
        # 执行阶段二
        stage2_result = self.stage2_algorithm.solve(self.network, stage1_result, flows)
        
        # 计算总执行时间
        total_execution_time = time.time() - start_time
        
        # 创建并返回完整结果
        return MonPlanResult(
            flows=flows,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
            total_execution_time=total_execution_time
        )
    
    def optimize_stage1_only(self, flows: Optional[List[Flow]] = None) -> Stage1Result:
        """仅执行阶段一优化
        
        Args:
            flows: 网络流列表，如果为None则自动生成
            
        Returns:
            阶段一结果
        """
        if self.stage1_algorithm is None:
            raise ValueError("阶段一算法未设置")
        
        if flows is None:
            flows = self.flow_generator.generate_flows()
        
        return self.stage1_algorithm.solve(self.network, flows)


if __name__ == "__main__":
    network = FatTree(k=4)
    network.generate_topology()
    network.add_control_nodes()
    
    # 创建流生成器和算法
    flow_generator = FlowGenerator(network, num_flows=50)
    stage1_algorithm = MonPlanStage1(verbose=True)
    
    # 创建MonPlan实例
    monplan = MonPlan(flow_generator, stage1_algorithm)
    
    # 仅执行阶段一
    stage1_result = monplan.optimize_stage1_only()
    print("阶段一结果:")
    print(f"流覆盖率: {stage1_result.flow_coverage:.3f}")
    print(f"使用交换机数: {stage1_result.total_switches_used}")
    print(f"总控制跳数: {stage1_result.total_control_hops}")
    
    # 导入阶段二算法
    from data_collection_routing import EscalaStage2
    monplan.set_stage2_algorithm(EscalaStage2())
    
    # 执行完整优化
    result = monplan.optimize(num_flows=50)
    print("\n完整优化结果:")
    print(f"流覆盖率: {result.stage1_result.flow_coverage:.3f}")
    print(f"拥塞率: {result.stage2_result.congestion_rate:.3f}")
    print(f"总执行时间: {result.total_execution_time:.2f}秒")