from typing import Tuple, List
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from netembd.network.network import Network

class Wan(Network):
    """广域网拓扑实现类，支持可编程节点和控制节点"""

    def __init__(self):
        super().__init__()

    def update_programmable_and_control_nodes(self):
        """生成广域网拓扑，设置所有节点为可编程，并添加控制节点"""
        self._set_all_nodes_programmable()
        self._add_optimal_control_nodes()

    def _set_all_nodes_programmable(self):
        """设置所有节点为可编程"""
        for node in self.get_nodes():
            self.set_node_programmable(node, True)
            
    def calculate_node_latencies(self, node_id: int) -> Tuple[float, float]:
        """计算指定节点的最差延迟和平均延迟
        
        Args:
            node_id: 节点ID
            
        Returns:
            Tuple[float, float]: (最差延迟, 平均延迟)
        """
        latencies = []
        for other_node in self.get_nodes():
            if other_node != node_id:
                path = self.get_shortest_path(other_node, node_id)
                if path:
                    latency = self.get_path_latency(path)
                    latencies.append(latency)
        
        if not latencies:
            return float('inf'), float('inf')
        
        return max(latencies), sum(latencies) / len(latencies)
    
    def find_optimal_control_locations(self) -> Tuple[int, int]:
        """找到最优的控制节点放置位置
        
        Returns:
            Tuple[int, int]: (最小最差延迟节点, 最小平均延迟节点)
        """
        worst_latencies = {}
        avg_latencies = {}
        
        for node in self.get_nodes():
            worst_latency, avg_latency = self.calculate_node_latencies(node)
            worst_latencies[node] = worst_latency
            avg_latencies[node] = avg_latency
        
        return (
            min(worst_latencies.items(), key=lambda x: x[1])[0],
            min(avg_latencies.items(), key=lambda x: x[1])[0]
        )
    
    def _add_optimal_control_nodes(self) -> Tuple[int, int]:
        """添加两个控制节点到最优位置
        
        Returns:
            Tuple[int, int]: (控制节点1 ID, 控制节点2 ID)
        """
        optimal_loc1, optimal_loc2 = self.find_optimal_control_locations()
        control_node1 = self.add_control_node(optimal_loc1)
        control_node2 = self.add_control_node(optimal_loc2)
        return control_node1, control_node2            

    def visualize(self):
        """可视化网络拓扑，支持可编程节点和控制节点的显示"""
        pos = nx.spring_layout(self._graph)
        
        # 对节点进行分类
        node_categories = {
            'control': [],
            'programmable': [],
            'normal': []
        }
        
        for node in self._graph.nodes():
            if self._node_resources[node].control_node:
                node_categories['control'].append(node)
            elif self._node_resources[node].programmable:
                node_categories['programmable'].append(node)
            else:
                node_categories['normal'].append(node)
        
        # 绘制不同类型的节点
        node_styles = {
            'normal': {'color': 'lightgray', 'size': 500, 'shape': 'o'},
            'programmable': {'color': '#32CD32', 'size': 500, 'shape': 's'},
            'control': {'color': '#FFD700', 'size': 600, 'shape': 'd'}
        }
        
        for category, nodes in node_categories.items():
            if nodes:  # 只绘制非空类别
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    nodelist=nodes,
                    node_color=node_styles[category]['color'],
                    node_size=node_styles[category]['size'],
                    node_shape=node_styles[category]['shape']
                )
        
        # 绘制边和标签
        nx.draw_networkx_edges(self._graph, pos, edge_color='gray', width=1, alpha=0.5)
        nx.draw_networkx_labels(self._graph, pos, font_size=8)
        
        # 添加边的延迟标签
        edge_labels = nx.get_edge_attributes(self._graph, 'latency')
        edge_labels = {k: f'{v:.1f}ms' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels, font_size=6)
        
        # 添加图例
        legend_elements = [
            Patch(facecolor='lightgray', label='normal'),
            Patch(facecolor='#32CD32', label='programmable'),
            Patch(facecolor='#FFD700', label='control')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Network Topology')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    wan = Wan()
    wan.load_nodes(r"C:\Users\czd\OneDrive\code\netembd\input\topology\wan\Internet2\nodes.csv")
    wan.load_edges(r"C:\Users\czd\OneDrive\code\netembd\input\topology\wan\Internet2\edges.csv")
    wan.update_programmable_and_control_nodes()
    # 可视化
    print(len(wan.get_nodes()))
    wan.visualize()