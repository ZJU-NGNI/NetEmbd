import csv
import random
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from netembd.interfaces.base_network import BaseNetwork, NodeResource
from netembd.network import Network, flow_generator

class FatTree(Network):
    def __init__(self, pod_num: int=4, generate_flag: bool=True, single_node_resources: NodeResource=None, edge_bandwidth: float=10000, latency_range: Tuple[float, float]=(1, 10)):
        """初始化FatTree网络
        
        Args:
            pod_num: Pod数量，必须为偶数
            generate_flag: 是否自动生成网络
            node_resources: 节点资源配置，格式为{'alu': x, 'stage': y, 'sram': z}
            edge_bandwidth: 边的带宽（Mbps）
            edge_latency: 边的延迟（ms）
        """
        super().__init__()
        self._pod_num = pod_num
        self._single_node_resources = single_node_resources or NodeResource(alu=16, stage=4, sram=2048, programmable=False, control_node=False)
        self._edge_bandwidth = edge_bandwidth
        self._latency_range = latency_range
        
        if generate_flag:
            if pod_num % 2 != 0 or pod_num < 0:
                raise ValueError("pod_num must be even number")
            self._generate_fat_tree()
            self.set_random_programmable_nodes()
            self.add_pod_control_nodes()
    
    def _generate_fat_tree(self):
        """生成FatTree网络
        
        根据pod_num生成FatTree拓扑，并初始化节点资源和边的属性
        """
        k = self._pod_num
        # 计算各层节点数量并保存为类属性
        self._core_switches = (k//2)**2  # 核心层交换机数量
        self._aggr_switches = k * k//2   # 汇聚层交换机数量
        self._edge_switches = k * k//2   # 边缘层交换机数量
        
        # 添加节点和初始化资源
        total_nodes = self._core_switches + self._aggr_switches + self._edge_switches
        for i in range(total_nodes):
            self._graph.add_node(i)
            # 初始化节点资源
            self._node_resources[i] = NodeResource(
                alu=self._single_node_resources.alu,
                stage=self._single_node_resources.stage,
                sram=self._single_node_resources.sram,
                programmable=False,
                control_node=False
            )
        
        # 添加边并设置属性
        # 1. 连接核心层和汇聚层
        pod_size = k//2
        for pod in range(k):
            for j in range(pod_size):
                for r in range(pod_size):
                    core = r * pod_size + j
                    aggr = self._core_switches + pod * pod_size + j
                    self._graph.add_edge(
                        core,
                        aggr,
                        bandwidth=self._edge_bandwidth,
                        latency=np.random.uniform(self._latency_range[0], self._latency_range[1])
                    )
        
        # 2. 连接汇聚层和边缘层
        for pod in range(k):
            for aggr in range(pod_size):
                for edge in range(pod_size):
                    aggr_switch = self._core_switches + pod * pod_size + aggr
                    edge_switch = self._core_switches + self._aggr_switches + pod * pod_size + edge
                    self._graph.add_edge(
                        aggr_switch,
                        edge_switch,
                        bandwidth=self._edge_bandwidth,
                        latency=np.random.uniform(self._latency_range[0], self._latency_range[1])
                    )

    def add_pod_control_nodes(self) -> List[int]:
        """为每个pod添加一个控制节点，并连接到该pod的所有汇聚层交换机
        
        Returns:
            List[int]: 添加的控制节点ID列表
        """
        k = self._pod_num
        pod_size = k//2
        control_nodes = []
        
        for pod in range(k):
            # 获取该pod的所有汇聚层交换机
            aggr_switches = [self._core_switches + pod * pod_size + aggr for aggr in range(pod_size)]
            # 选择该pod中延迟最小的汇聚层交换机作为控制节点的连接点
            # 添加控制节点并连接
            control_node = self.add_control_node(aggr_switches[0])
            control_nodes.append(control_node)
            
        return control_nodes

    
    def set_random_programmable_nodes(self, select_ratio: float = 0.3) -> None:
        """随机选择指定比例的交换机设置为可编程
        
        Args:
            select_ratio: 设置为可编程的节点比例，默认为0.3
        """
        import random
        total_switches = self._core_switches + self._aggr_switches + self._edge_switches
        num_programmable = int(total_switches * select_ratio)
        programmable_nodes = random.sample(list(range(total_switches)), num_programmable)
        for node in programmable_nodes:
            self.set_node_programmable(node, True)

    def visualize(self):
        """以层级布局可视化FatTree网络拓扑，包括控制节点和可编程节点的显示
        
        使用不同的颜色和形状来区分节点类型：
        - 普通节点：蓝色圆形
        - 可编程节点：绿色方形
        - 控制节点：红色三角形
        """
        k = self._pod_num
        core_switches = (k//2)**2
        aggr_switches = k * k//2
        edge_switches = k * k//2

        # 计算每层节点的位置
        pos = {}
        
        # 1. 布局核心层交换机
        core_x_step = 1.0 / (core_switches + 1)
        for i in range(core_switches):
            pos[i] = (core_x_step * (i + 1), 0.9)
        # 2. 布局汇聚层交换机
        aggr_x_step = 1.0 / (aggr_switches + 1)
        for i in range(aggr_switches):
            pos[core_switches + i] = (aggr_x_step * (i + 1), 0.6)
        
        # 3. 布局边缘层交换机
        edge_x_step = 1.0 / (edge_switches + 1)
        for i in range(edge_switches):
            pos[core_switches + aggr_switches + i] = (edge_x_step * (i + 1), 0.3)
        
        # 4. 布局控制节点（在各自连接的交换机附近）
        control_nodes = [n for n in self._graph.nodes() if self._node_resources[n].control_node]
        for control_node in control_nodes:
            # 获取控制节点连接的交换机
            neighbors = list(self._graph.neighbors(control_node))
            if neighbors:
                # 将控制节点放在连接交换机的上方
                switch_pos = pos[neighbors[0]]
                pos[control_node] = (switch_pos[0], switch_pos[1] + 0.1)
        
        # 绘制节点
        plt.figure(figsize=(12, 8))
        
        # 绘制不同类型的节点
        normal_nodes = [n for n in self._graph.nodes() if not self._node_resources[n].programmable and not self._node_resources[n].control_node]
        programmable_nodes = [n for n in self._graph.nodes() if self._node_resources[n].programmable]
        
        # 绘制普通节点（蓝色圆形）
        nx.draw_networkx_nodes(self._graph, pos, nodelist=normal_nodes, node_color='lightblue', node_size=500)
        
        # 绘制可编程节点（绿色方形）
        nx.draw_networkx_nodes(self._graph, pos, nodelist=programmable_nodes, node_color='lightgreen', node_shape='s', node_size=500)
        
        # 绘制控制节点（红色三角形）
        nx.draw_networkx_nodes(self._graph, pos, nodelist=control_nodes, node_color='red', node_shape='^', node_size=500)
        
        # 绘制边
        nx.draw_networkx_edges(self._graph, pos)
        
        # 添加节点标签
        nx.draw_networkx_labels(self._graph, pos)
        
        # 添加图例
        from matplotlib.patches import Patch, Circle, RegularPolygon
        legend_elements = [
            Circle((0, 0), 1, facecolor='lightblue', label='普通节点'),
            RegularPolygon((0, 0), 4, facecolor='lightgreen', label='可编程节点'),
            RegularPolygon((0, 0), 3, facecolor='red', label='控制节点')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f'FatTree Network (k={self._pod_num})')
        plt.axis('off')
        plt.tight_layout()
        
        plt.show()
    
    def generate_flows(self, num_flows: int, flow_size_range: Tuple[float, float] = (1.0, 1000.0), 
                      large_flow_threshold: float = 100.0) -> List['Flow']:
        """生成DCN网络流（OD pair均为叶子节点）
        
        Args:
            num_flows: 生成的流数量
            flow_size_range: 流大小范围
            large_flow_threshold: 大流阈值，超过此值使用sketch测量
            
        Returns:
            DCN流列表
        """
        from netembd.network.flow_generator import Flow
        self.flow_generator = self.create_flow_generator()
        self.flows = []
        leaf_nodes = self._get_leaf_nodes()
        
        if len(leaf_nodes) < 2:
            return flows
        
        for i in range(num_flows):
            # 随机选择两个不同的叶子节点作为OD pair
            origin, destination = random.sample(leaf_nodes, 2)
            
            # 生成随机流大小
            flow_size = random.uniform(*flow_size_range)
            
            flow = self.flow_generator.generate_a_flow(flow_id=i, origin=origin, destination=destination, size=flow_size)
            
            if flow is None:
                continue
            
            self.flows.append(flow)

        return self.flows
    
    def _get_leaf_nodes(self) -> List[int]:
        """获取DCN网络的叶子节点（边缘交换机）
        
        Returns:
            叶子节点ID列表
        """
        # FatTree的叶子节点是边缘交换机
        core_switches = self._core_switches
        aggr_switches = self._aggr_switches
        start_edge = core_switches + aggr_switches
        end_edge = start_edge + self._edge_switches
        return list(range(start_edge, end_edge))

if __name__ == '__main__':
    ft = FatTree(pod_num=48)
    # ft.visualize()
    print(len(ft.get_nodes()))
