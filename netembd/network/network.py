"""网络拓扑实现模块。

此模块实现了网络拓扑的具体功能，包括：
1. 从CSV文件加载节点和边的配置
2. 使用NetworkX库管理网络结构
3. 计算最短路径和延迟
4. 管理节点资源

Typical usage example:

    from netembd.network import Network
    
    network = Network()
    network.load_nodes("nodes.csv")
    network.load_edges("edges.csv")
"""

import csv
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from netembd.interfaces.base_network import BaseNetwork, NodeResource

class NetworkVisualizationMixin:
    """网络可视化混入类，提供基础的可视化功能"""
    
    def visualize(self):
        """可视化网络拓扑"""
        pos = nx.spring_layout(self._graph)
        nx.draw(self._graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)
        edge_labels = nx.get_edge_attributes(self._graph, 'latency')
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=edge_labels, font_size=8)
        plt.show()

class ProgrammableNetworkMixin:
    """可编程网络混入类，提供节点可编程性管理功能"""
    
    def set_node_programmable(self, node_id: int, programmable: bool = True) -> None:
        """设置节点的可编程性
        
        Args:
            node_id: 节点ID
            programmable: 是否可编程
        """
        if node_id not in self._node_resources:
            raise ValueError(f"节点{node_id}不存在")
        self._node_resources[node_id].programmable = programmable
    
    def is_node_programmable(self, node_id: int) -> bool:
        """检查节点是否可编程
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否可编程
        """
        if node_id not in self._node_resources:
            raise ValueError(f"节点{node_id}不存在")
        return self._node_resources[node_id].programmable

class ControlNodeMixin:
    """控制节点混入类，提供控制节点管理功能"""
    
    def add_control_node(self, target_node: int) -> int:
        """添加控制节点并连接到目标节点
        
        Args:
            target_node: 目标节点ID
            
        Returns:
            控制节点ID
        """
        if target_node not in self._node_resources:
            raise ValueError(f"目标节点{target_node}不存在")
            
        control_node = max(self._graph.nodes()) + 1
        self._graph.add_node(control_node)
        self._node_resources[control_node] = NodeResource(
            alu=0,
            stage=0,
            sram=0,
            programmable=False,
            control_node=True
        )
        self._graph.add_edge(control_node, target_node, latency=1.0, bandwidth=1000.0)
        self._shortest_paths.clear()
        return control_node

class Network(BaseNetwork, NetworkVisualizationMixin, ProgrammableNetworkMixin, ControlNodeMixin):
    """网络拓扑基础实现类"""
    
    def __init__(self):
        """初始化网络拓扑"""
        self._graph = nx.Graph()
        self._node_resources: Dict[int, NodeResource] = {}
        self._shortest_paths: Dict[Tuple[int, int], List[int]] = {}
    
    def load_nodes(self, node_csv: str) -> None:
        """从CSV文件加载节点配置
        
        Args:
            node_csv: 节点配置CSV文件路径
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        try:
            with open(node_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node_id = int(row['node_id'])
                    self._node_resources[node_id] = NodeResource(
                        alu=int(row['alu']),
                        stage=int(row['stage']),
                        sram=int(row['sram'])
                    )
                    self._graph.add_node(node_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"节点配置文件不存在：{node_csv}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"节点配置文件格式错误：{str(e)}")
    
    def load_edges(self, edge_csv: str) -> None:
        """从CSV文件加载边配置
        
        Args:
            edge_csv: 边配置CSV文件路径
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        try:
            with open(edge_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source = int(row['source'])
                    target = int(row['target'])
                    self._graph.add_edge(
                        source,
                        target,
                        bandwidth=float(row['bandwidth']),
                        latency=float(row['latency'])
                    )
            # 清空最短路径缓存
            self._shortest_paths.clear()
        except FileNotFoundError:
            raise FileNotFoundError(f"边配置文件不存在：{edge_csv}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"边配置文件格式错误：{str(e)}")
    
    def get_nodes(self) -> List[int]:
        """获取所有节点ID
        
        Returns:
            节点ID列表
        """
        return list(self._graph.nodes())
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """获取所有边
        
        Returns:
            边列表，每条边为(源节点ID, 目标节点ID)元组
        """
        return list(self._graph.edges())
    
    def get_node_resource(self, node_id: int) -> NodeResource:
        """获取节点的资源信息
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点资源对象
            
        Raises:
            ValueError: 节点不存在
        """
        if node_id not in self._node_resources:
            raise ValueError(f"节点{node_id}不存在")
        return self._node_resources[node_id]
    
    def get_edge_latency(self, source: int, target: int) -> Optional[float]:
        """获取边的通信延迟
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            通信延迟（毫秒），如果边不存在则返回None
        """
        try:
            return self._graph[source][target]['latency']
        except KeyError:
            return None
    
    def get_edge_bandwidth(self, source: int, target: int) -> Optional[float]:
        """获取边的带宽
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            带宽（Mbps），如果边不存在则返回None
        """
        try:
            return self._graph[source][target]['bandwidth']
        except KeyError:
            return None
    
    def get_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """计算两个节点间的最短路径
        
        使用Dijkstra算法计算基于延迟的最短路径。结果会被缓存以提高性能。
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            节点ID列表表示的路径，如果不存在路径则返回None
        """
        if (source, target) not in self._shortest_paths:
            try:
                path = nx.shortest_path(
                    self._graph,
                    source=source,
                    target=target,
                    weight='latency'
                )
                self._shortest_paths[(source, target)] = path
            except nx.NetworkXNoPath:
                return None
        return self._shortest_paths[(source, target)]
    
    def get_path_latency(self, path: List[int]) -> float:
        """计算路径的总延迟
        
        Args:
            path: 节点ID列表表示的路径
        
        Returns:
            路径总延迟（毫秒）
        
        Raises:
            ValueError: 路径无效
        """
        if len(path) < 2:
            raise ValueError("路径至少需要包含两个节点")
        
        total_latency = 0.0
        for i in range(len(path) - 1):
            latency = self.get_edge_latency(path[i], path[i + 1])
            if latency is None:
                raise ValueError(f"节点{path[i]}和{path[i + 1]}之间不存在边")
            total_latency += latency
        return total_latency
    
    def generate_flows(self, num_flows: int, flow_size_range: Tuple[float, float] = (1.0, 1000.0), 
                      large_flow_threshold: float = 100.0) -> List['Flow']:
        """生成网络流的默认实现
        
        Args:
            num_flows: 生成的流数量
            flow_size_range: 流大小范围
            large_flow_threshold: 大流阈值，超过此值使用sketch测量
            
        Returns:
            流列表
        """
        import random
        from .flow_generator import Flow
        
        flows = []
        non_control_nodes = self._get_non_control_nodes()
        
        if len(non_control_nodes) < 2:
            return flows
            
        for i in range(num_flows):
            # 随机选择两个不同的非控制节点作为OD pair
            origin, destination = random.sample(non_control_nodes, 2)
            
            # 计算最短路径
            path = self.get_shortest_path(origin, destination)
            if path is None:
                continue
            
            # 生成随机流大小
            flow_size = random.uniform(*flow_size_range)
            
            # 分配测量类型
            measurement_type = 'sketch' if flow_size >= large_flow_threshold else 'INT'
            
            flow = Flow(
                flow_id=i,
                origin=origin,
                destination=destination,
                size=flow_size,
                path=path,
                measurement_type=measurement_type
            )
            flows.append(flow)
        self.flows = flows
        return flows
    
    def _get_non_control_nodes(self) -> List[int]:
        """获取非控制节点
        
        Returns:
            非控制节点ID列表
        """
        non_control_nodes = []
        for node in self.get_nodes():
            resource = self.get_node_resource(node)
            if not resource.control_node:
                non_control_nodes.append(node)
        return non_control_nodes
    
    def create_flow_generator(self, num_flows: int = 100, 
                             flow_size_range: tuple = (1.0, 1000.0),
                             large_flow_threshold: float = 100.0):
        """创建流生成器的便捷方法
        
        Args:
            num_flows: 默认生成的流数量
            flow_size_range: 流大小范围
            large_flow_threshold: 大流阈值
            
        Returns:
            FlowGenerator实例
        """
        from .flow_generator import FlowGenerator
        return FlowGenerator(self, num_flows, flow_size_range, large_flow_threshold)






