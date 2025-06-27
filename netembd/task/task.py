"""任务DAG实现模块。

此模块实现了任务DAG的具体功能，包括：
1. 从CSV文件加载VNF和依赖关系
2. 使用NetworkX库管理DAG结构
3. 计算关键路径
4. 管理VNF资源需求和依赖关系

Typical usage example:

    from netembd.task import Task
    
    task = Task()
    task.load_vnfs("vnfs.csv")
    task.load_dependencies("deps.csv")
"""

import csv
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from netembd.interfaces.base_task import BaseTask, VNFResource, VNFDependency

class Task(BaseTask):
    """任务DAG实现类"""
    
    def __init__(self):
        """初始化任务DAG"""
        self._graph = nx.DiGraph()
        self._vnf_resources: Dict[int, VNFResource] = {}
        self._vnf_dependencies: Dict[Tuple[int, int], VNFDependency] = {}
        self._critical_path: Optional[List[int]] = None
    
    def load_vnfs(self, vnf_csv: str) -> None:
        """从CSV文件加载VNF配置
        
        Args:
            vnf_csv: VNF配置CSV文件路径
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        try:
            with open(vnf_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vnf_id = int(row['vnf_id'])
                    self._vnf_resources[vnf_id] = VNFResource(
                        alu=int(row['alu']),
                        stage=int(row['stage']),
                        sram=int(row['sram'])
                    )
                    self._graph.add_node(vnf_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"VNF配置文件不存在：{vnf_csv}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"VNF配置文件格式错误：{str(e)}")
    
    def load_dependencies(self, dep_csv: str) -> None:
        """从CSV文件加载VNF依赖关系
        
        Args:
            dep_csv: 依赖关系CSV文件路径
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误或存在循环依赖
        """
        try:
            with open(dep_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source = int(row['source'])
                    target = int(row['target'])
                    dep = VNFDependency(
                        source,
                        target,
                        data_size=float(row['data_size'])
                    )
                    self._vnf_dependencies[(source, target)] = dep
                    self._graph.add_edge(source, target)
            
            # 检查是否存在循环依赖
            try:
                nx.find_cycle(self._graph)
                raise ValueError("存在循环依赖")
            except nx.NetworkXNoCycle:
                pass
            
            # 清空关键路径缓存
            self._critical_path = None
        except FileNotFoundError:
            raise FileNotFoundError(f"依赖关系配置文件不存在：{dep_csv}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"依赖关系配置文件格式错误：{str(e)}")
    
    def get_vnfs(self) -> List[int]:
        """获取所有VNF ID
        
        Returns:
            VNF ID列表
        """
        return list(self._graph.nodes())
    
    def get_dependencies(self) -> List[Tuple[int, int]]:
        """获取所有依赖关系
        
        Returns:
            依赖关系列表，每个依赖为(源VNF ID, 目标VNF ID)元组
        """
        return list(self._graph.edges())
    
    def get_vnf_resource(self, vnf_id: int) -> VNFResource:
        """获取VNF的资源需求
        
        Args:
            vnf_id: VNF ID
            
        Returns:
            VNF资源需求对象
            
        Raises:
            ValueError: VNF不存在
        """
        if vnf_id not in self._vnf_resources:
            raise ValueError(f"VNF {vnf_id} 不存在")
        return self._vnf_resources[vnf_id]
    
    def get_dependency_data_size(self, source: int, target: int) -> Optional[float]:
        """获取依赖关系的数据传输量
        
        Args:
            source: 源VNF ID
            target: 目标VNF ID
        
        Returns:
            数据传输量（MB），如果依赖不存在则返回None
        """
        dep = self._vnf_dependencies.get((source, target))
        return dep.data_size if dep else None
    
    def get_predecessors(self, vnf_id: int) -> Set[int]:
        """获取VNF的前驱节点
        
        Args:
            vnf_id: VNF ID
        
        Returns:
            前驱VNF ID集合
        """
        return set(self._graph.predecessors(vnf_id))
    
    def get_successors(self, vnf_id: int) -> Set[int]:
        """获取VNF的后继节点
        
        Args:
            vnf_id: VNF ID
        
        Returns:
            后继VNF ID集合
        """
        return set(self._graph.successors(vnf_id))
    
    def get_critical_path(self) -> List[int]:
        """计算关键路径
        
        使用拓扑排序和动态规划计算最长路径。结果会被缓存以提高性能。
        
        Returns:
            VNF ID列表表示的关键路径
        """
        if self._critical_path is None:
            # 拓扑排序
            topo_order = list(nx.topological_sort(self._graph))
            
            # 动态规划计算最长路径
            path_length: Dict[int, float] = {}
            prev_node: Dict[int, Optional[int]] = {}
            
            # 初始化
            for vnf_id in topo_order:
                path_length[vnf_id] = 0.0
                prev_node[vnf_id] = None
            
            # 计算每个节点的最长路径
            for vnf_id in topo_order:
                for succ in self.get_successors(vnf_id):
                    data_size = self.get_dependency_data_size(vnf_id, succ)
                    new_length = path_length[vnf_id] + data_size
                    if new_length > path_length[succ]:
                        path_length[succ] = new_length
                        prev_node[succ] = vnf_id
            
            # 找到终点（出度为0的节点）中路径最长的
            end_nodes = [n for n in self._graph.nodes() if self._graph.out_degree(n) == 0]
            end_node = max(end_nodes, key=lambda n: path_length[n])
            
            # 回溯构建关键路径
            path = []
            curr = end_node
            while curr is not None:
                path.append(curr)
                curr = prev_node[curr]
            
            self._critical_path = list(reversed(path))
        
        return self._critical_path.copy()
    
    def visualize(self) -> None:
        """可视化任务DAG
        
        使用NetworkX和Matplotlib绘制任务DAG图，包括：
        1. 节点：显示VNF ID和资源需求
        2. 边：显示数据传输量
        3. 关键路径：使用不同颜色高亮显示
        
        Raises:
            ImportError: 缺少必要的可视化库
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("请安装matplotlib库以支持可视化功能")
        
        # 计算关键路径
        critical_path = self.get_critical_path()
        critical_edges = list(zip(critical_path[:-1], critical_path[1:]))
        
        # 设置绘图样式
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self._graph)
        
        # 绘制边（非关键路径）
        nx.draw_networkx_edges(
            self._graph,
            pos,
            edgelist=[e for e in self._graph.edges() if e not in critical_edges],
            edge_color='gray',
            arrows=True
        )
        
        # 绘制关键路径边
        if critical_edges:
            nx.draw_networkx_edges(
                self._graph,
                pos,
                edgelist=critical_edges,
                edge_color='red',
                arrows=True,
                width=2
            )
        
        # 绘制节点
        nx.draw_networkx_nodes(
            self._graph,
            pos,
            node_color=['red' if n in critical_path else 'lightblue' for n in self._graph.nodes()],
            node_size=1000
        )
        
        # 添加节点标签
        labels = {}
        for n in self._graph.nodes():
            resource = self.get_vnf_resource(n)
            labels[n] = f'VNF {n}\nALU: {resource.alu}\nStage: {resource.stage}\nSRAM: {resource.sram}'
        nx.draw_networkx_labels(self._graph, pos, labels)
        
        # 添加边标签
        edge_labels = {}
        for s, t in self._graph.edges():
            data_size = self.get_dependency_data_size(s, t)
            edge_labels[(s, t)] = f'{data_size:.1f} MB'
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels)
        
        plt.title('任务DAG可视化（红色表示关键路径）')
        plt.axis('off')
        plt.show()