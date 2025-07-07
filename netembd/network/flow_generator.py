"""网络流生成器模块。

此模块实现了网络流生成功能，包括：
1. 基于网络拓扑的流生成
2. 流大小和测量类型的自动分配
3. 作为网络类generate_flows方法的便捷包装器

Typical usage example:

    from netembd.network.flow_generator import FlowGenerator
    from netembd.network import Network
    
    network = Network()
    flow_generator = FlowGenerator(network, num_flows=100)
    flows = flow_generator.generate_flows()
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from netembd.interfaces.base_network import BaseNetwork

@dataclass
class Flow:
    """网络流数据类"""
    flow_id: int
    origin: int
    destination: int
    size: float  # 流大小，用于区分大流和小流
    path: List[int]  # 流经路径
    measurement_type: Optional[str] = None  # 'sketch' 或 'INT'

class FlowGenerator:
    """网络流生成器
    
    作为网络类generate_flows方法的便捷包装器，提供统一的流生成接口。
    具体的流生成逻辑由各个网络类自己实现。
    """
    
    def __init__(self, network: BaseNetwork, num_flows: int = 100, 
                 flow_size_range: Tuple[float, float] = (1.0, 1000.0),
                 large_flow_threshold: float = 100.0):
        """初始化流生成器
        
        Args:
            network: 网络拓扑实例
            num_flows: 默认生成的流数量
            flow_size_range: 流大小范围
            large_flow_threshold: 大流阈值，超过此值使用sketch测量
        """
        self.network = network
        self.num_flows = num_flows
        self.flow_size_range = flow_size_range
        self.large_flow_threshold = large_flow_threshold
        self.available_bandwidth: Dict[Tuple[int, int], float] = {}
        self.initialize_available_bandwidth()
        
    def initialize_available_bandwidth(self):
        """初始化每个链路的可用带宽"""
        for link in self.network.get_links():
            # 存储双向链路，因为NetworkX的Graph是无向图
            link_key_1 = (link.source, link.target)
            link_key_2 = (link.target, link.source)
            self.available_bandwidth[link_key_1] = link.bandwidth
            self.available_bandwidth[link_key_2] = link.bandwidth

    def update_available_bandwidth(self, flow: Flow, decrease: bool = True):
        """更新可用带宽
        Args:
            flow: 流
            decrease: 是否减少可用带宽
        """
        for i in range(len(flow.path) - 1):
            link = (flow.path[i], flow.path[i + 1])
            if decrease:
                self.available_bandwidth[link] -= flow.size
            else:
                self.available_bandwidth[link] += flow.size

    def find_path(self, origin: int, destination: int, flow_size: float) -> List[int]:
        """查找一条满足流量需求的路径
        
        Args:
            origin: 流源
            destination: 流目的
            flow_size: 流大小
        
        Returns:
            满足流量需求的路径
        """
        # 简单实现：直接返回最短路径
        shortest_path = self.network.get_shortest_path(origin, destination)
        # 检查路径上的可用带宽是否足够
        for i in range(len(shortest_path) - 1):
            link = (shortest_path[i], shortest_path[i + 1])
            if self.available_bandwidth[link] < flow_size:
                return None  # 路径不可行
        return shortest_path

    def generate_a_flow(self, flow_id: int, origin: int, destination: int, size: float, measurement_type: Optional[str] = None) -> Flow:
        """生成一个网络流
        
        Args:
            flow_id: 流ID
            origin: 流源
            destination: 流目的
            size: 流大小
            measurement_type: 测量类型
        
        Returns:
            生成的流
        """
        flow_path = self.find_path(origin, destination, size)
        if flow_path is None:
            return None
        if measurement_type is None:
            if size >= self.large_flow_threshold:
                measurement_type = 'sketch'
            else:
                measurement_type = 'INT'

        flow = Flow(
            flow_id=flow_id,
            origin=origin,
            destination=destination,
            size=size,
            measurement_type=measurement_type,
            path = flow_path
        )
        self.update_available_bandwidth(flow)
        return flow

    def generate_flows(self, num_flows: Optional[int] = None) -> List[Flow]:
        """生成网络流
        
        Args:
            num_flows: 生成的流数量，如果为None则使用初始化时的值
        
        Returns:
            生成的流列表
        """
        if num_flows is None:
            num_flows = self.num_flows
        
        # 委托给网络类的generate_flows方法
        return self.network.generate_flows(
            num_flows=num_flows,
            flow_size_range=self.flow_size_range,
            large_flow_threshold=self.large_flow_threshold
        )