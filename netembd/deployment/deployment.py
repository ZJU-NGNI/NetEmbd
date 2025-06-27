"""部署方案实现模块。

此模块实现了部署方案的具体功能，包括：
1. 管理VNF到物理节点的映射
2. 计算资源使用情况
3. 验证部署方案的可行性
4. 提供CSV导入导出和可视化功能

Typical usage example:

    from netembd.deployment import Deployment
    from netembd.network import Network
    from netembd.task import Task
    
    deployment = Deployment(network, task)
    deployment.assign_vnf(vnf_id=0, node_id=1)
    deployment.save_to_csv("deployment.csv")
"""

from argparse import OPTIONAL
import csv
from typing import Dict, List, Optional, Set, Tuple, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from netembd.interfaces.base_deployment import BaseDeployment, Assignment
from netembd.interfaces.base_network import BaseNetwork
from netembd.interfaces.base_task import BaseTask

class Deployment(BaseDeployment):
    """部署方案实现类"""
    
    def __init__(self, network: BaseNetwork, task: BaseTask):
        """初始化部署方案
        
        Args:
            network: 网络拓扑对象
            task: 任务DAG对象
        """
        self._network = network
        self._task = task
        self._assignments: Dict[int, Assignment] = {}
        self._node_vnfs: Dict[int, Set[int]] = {}
        for node_id in network.get_nodes():
            self._node_vnfs[node_id] = set()
    
    def assign_vnf(self, vnf_id: int, node_id: int, stage_id: int) -> None:
        """将VNF分配到物理节点
        
        Args:
            vnf_id: VNF ID
            node_id: 物理节点ID
            stage_id: VNF所在的stage ID
        
        Raises:
            ValueError: VNF或节点不存在
        """
        # 检查VNF和节点是否存在
        if vnf_id not in self._task.get_vnfs():
            raise ValueError(f"VNF不存在：{vnf_id}")
        if node_id not in self._network.get_nodes():
            raise ValueError(f"节点不存在：{node_id}")
        if stage_id < 0 or stage_id >= self._network.get_node_resource(node_id).stage:
            raise ValueError(f"stage ID 超出范围：{stage_id}")
        # 如果VNF已分配，从原节点移除
        old_assignment = self._assignments.get(vnf_id)
        if old_assignment:
            self._node_vnfs[old_assignment.node_id].remove(vnf_id)
        
        # 添加新分配
        self._assignments[vnf_id] = Assignment(vnf_id=vnf_id, node_id=node_id, stage_id=stage_id)
        self._node_vnfs[node_id].add(vnf_id)
    
    def get_vnf_assignment(self, vnf_id: int) -> Optional[Assignment]:
        """获取VNF的分配信息
        
        Args:
            vnf_id: VNF ID
        
        Returns:
            分配信息对象，如果VNF未分配则返回异常
        """
        try:
            assignment = self._assignments.get(vnf_id)
            if assignment:
                return assignment
            else:
                raise ValueError(f"VNF未分配：{vnf_id}")
        except ValueError as e:
            raise ValueError(f"获取VNF分配信息出错：{str(e)}")
    
    def get_node_vnfs(self, node_id: int) -> Set[int]:
        """获取节点上分配的所有VNF
        
        Args:
            node_id: 物理节点ID
        
        Returns:
            VNF ID集合
        
        Raises:
            ValueError: 节点不存在
        """
        if node_id not in self._network.get_nodes():
            raise ValueError(f"节点不存在：{node_id}")
        return self._node_vnfs[node_id].copy()
    
    def calculate_node_resource_usage(self, node_id: int) -> Dict[str, float]:
        """计算节点的资源使用情况
        
        Args:
            node_id: 物理节点ID
        
        Returns:
            资源使用情况字典，包含'alu'、'stage'和'sram'的使用率
        
        Raises:
            ValueError: 节点不存在
        """
        node_resource = self._network.get_node_resource(node_id)
        if not node_resource:
            raise ValueError(f"节点不存在：{node_id}")
        
        total_alu = 0
        total_sram = 0
        
        for vnf_id in self._node_vnfs[node_id]:
            vnf_resource = self._task.get_vnf_resource(vnf_id)
            if vnf_resource:
                total_alu += vnf_resource.alu
                total_sram += vnf_resource.sram
        
        return {
            'alu': total_alu / node_resource.alu,
            'stage': len(self._node_vnfs[node_id]) / node_resource.stage,
            'sram': total_sram / node_resource.sram
        }



    def calculate_total_latency(self) -> float:
        """计算部署方案的总延迟
        
        Returns:
            总延迟（毫秒）
        
        Raises:
            ValueError: 存在未分配的VNF或无效的路径
        """
        total_latency = 0.0
        
        # 检查所有VNF是否都已分配
        for vnf_id in self._task.get_vnfs():
            if vnf_id not in self._assignments:
                raise ValueError(f"VNF未分配：{vnf_id}")
        
        # 计算所有依赖边的延迟
        for source, target in self._task.get_dependencies():
            source_node = self._assignments[source].node_id
            target_node = self._assignments[target].node_id
            if source_node == target_node:
                continue
            # 获取最短路径
            path = self._network.get_shortest_path(source_node, target_node)
            if not path:
                raise ValueError(f"节点{source_node}到{target_node}之间不存在路径")
            
            # 计算传输延迟
            data_size = self._task.get_dependency_data_size(source, target)
            path_latency = self._network.get_path_latency(path)
            # total_latency += path_latency * data_size
            total_latency += path_latency
        
        return total_latency
    
    def is_feasible(self) -> bool:
        """检查部署方案是否可行
        
        检查以下约束：
        1. 所有VNF都已分配
        2. 节点资源未超限
        3. 所有依赖边都有可行路径
        
        Returns:
            是否可行
        """
        try:
            # 检查所有VNF是否都已分配
            for vnf_id in self._task.get_vnfs():
                if vnf_id not in self._assignments:
                    return False
            
            # 检查节点资源是否超限
            for node_id in self._network.get_nodes():
                usage = self.calculate_node_resource_usage(node_id)
                if any(u > 1.0 for u in usage.values()):
                    return False
            
            # 检查所有依赖边是否有可行路径
            for source, target in self._task.get_dependencies():
                source_node = self._assignments[source].node_id
                target_node = self._assignments[target].node_id
                if not self._network.get_shortest_path(source_node, target_node):
                    return False
            
            return True
        except ValueError:
            return False
    
    def get_assignments(self) -> Dict[int, Assignment]:
        """获取所有VNF的分配信息
        
        Returns:
            VNF分配信息字典，键为VNF ID，值为分配信息对象
        """
        return self._assignments.copy()
    
    def save_to_csv(self, csv_path: str) -> None:
        """将部署方案保存到CSV文件
        
        Args:
            csv_path: CSV文件路径
        """
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vnf_id', 'node_id', "stage_id"])
            for vnf_id, assignment in sorted(self._assignments.items()):
                writer.writerow([vnf_id, assignment.node_id, assignment.stage_id])
    
    
    def visualize(self, output_path: str = None) -> None:
        """可视化部署方案
        
        生成一个2D热力图，展示VNF在各个交换机阶段上的部署情况
        - 纵坐标：VNF ID
        - 横坐标：Switch-Stage组合
        
        Args:
            output_path: 输出图片路径，如果为None则直接显示图形
        """
        # 获取所有VNF和它们的分配信息
        vnf_ids = sorted(self._task.get_vnfs())
        node_ids = sorted(self._network.get_nodes())
        
        # 生成Switch-Stage组合标签
        placement_labels = []
        for node_id in node_ids:
            for stage_id in range(self._network.get_node_resource(node_id).stage):
                placement_labels.append(f"{node_id}-{stage_id}")
        
        # 创建部署矩阵
        deployment_matrix = np.zeros((len(vnf_ids), len(placement_labels)))
        
        # 填充矩阵
        for i, vnf_id in enumerate(vnf_ids):
            if vnf_id in self._assignments:
                assignment = self._assignments[vnf_id]
                if assignment.stage_id is not None:
                    label_idx = placement_labels.index(f"{assignment.node_id}-{assignment.stage_id}")
                    deployment_matrix[i, label_idx] = 1
                else:
                    for stage_id in range(self._network.get_node_resource(assignment.node_id).stage):
                        label_idx = placement_labels.index(f"{assignment.node_id}-{stage_id}")
                        deployment_matrix[i, label_idx] = 1
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(deployment_matrix, cmap='binary', edgecolors='gray', linewidth=1)
        
        # 设置坐标轴标签
        plt.yticks(np.arange(0.5, len(vnf_ids), 1), [f"VNF {vid}" for vid in vnf_ids])
        plt.xticks(np.arange(0.5, len(placement_labels), 1), placement_labels, rotation=90)
        
        plt.xlabel("Switch-Stage")
        plt.ylabel("VNF ID")
        plt.title("VNF Deployment Map")
        
        # 添加网格线
        plt.grid(True, which='major', color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
        plt.grid(True, which='minor', color='lightgray', linestyle=':', alpha=0.2, linewidth=0.5)
        
        # 调整布局以适应标签
        plt.gca().set_axisbelow(True)
        
        # 调整布局以防止标签被截断
        plt.tight_layout()
        
        # 保存或显示图形
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()

        
