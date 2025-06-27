"""Hermes启发式算法实现模块。

此模块实现了基于任务分层和资源感知的Hermes启发式算法，包括：
1. 任务依赖图分割
2. 基于资源和延迟的交换机选择
3. 提供求解状态和结果查询

Typical usage example:

    from netembd.optimize.hermes import HermesHeuristic
    
    optimizer = HermesHeuristic(network, task, epsilon1=100000, epsilon2=100, k=1)
    deployment = optimizer.solve()
"""

import heapq
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from netembd.deployment import Deployment
from netembd.network import Network
from netembd.task import Task
from netembd.optimize.heuristic import HeuristicOptimizer
from netembd.interfaces.base_optimizer import OptimizerConfig, OptimizationStatus

class HermesHeuristic(HeuristicOptimizer):
    """Hermes启发式算法实现类"""
    
    def __init__(self, network: Network, task: Task, epsilon1: float = 100000,  epsilon2: int = 100, k: int = 1, config: Optional[OptimizerConfig] = None):
        """初始化Hermes启发式算法求解器
        
        Args:
            network: 网络拓扑对象
            task: 任务DAG对象
            epsilon1: 端到端传输延迟约束
            epsilon2: 最大占用节点数量约束
            k: 最短路径数量
            config: 优化器配置对象
        """
        super().__init__(network, task, config)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.k = k
        
        # 计算网络资源基准
        self.min_sram = min(network.get_node_resource(node).sram for node in network.get_nodes())
        self.min_stages = min(network.get_node_resource(node).stage for node in network.get_nodes())
    
    def solve(self) -> Optional[Deployment]:
        """求解VNF部署优化问题
        
        Returns:
            部署方案对象，如果未找到可行解则返回None
        """
        self._status = OptimizationStatus.RUNNING
        
        # 步骤1: 分割任务依赖图
        segments = self.split_tdg(self._task._graph)
        print(f"任务被分割为 {len(segments)} 个段")        
        # 步骤2: 寻找合适的交换机进行部署
        deployed = False

        for anchor_switch in self._network.get_nodes():
            # 选择候选交换机
            candidate_switches = self._select_switches(anchor_switch)
            print("candidate switch for anchor", anchor_switch, candidate_switches)
            # 检查是否有足够的交换机
            if len(candidate_switches) < len(segments):
                print("没有足够的交换机")
                continue
            self._status = OptimizationStatus.SOLVED
            self._best_deployment = Deployment(self._network, self._task)
            # 部署任务段到候选交换机
            for i, segment in enumerate(segments):
                self._deploy_segment(segment, candidate_switches[i])
            
            deployed = True
            print(f"任务成功部署在 {len(segments)} 个交换机上")
            break
        
        if not deployed:
            raise RuntimeError("无法找到满足约束的部署方案")
        
        return self._best_deployment        
    
    def split_tdg(self, graph: nx.DiGraph) -> List[nx.DiGraph]:
        """递归分割任务依赖图
        
        Args:
            graph: 要分割的图
            
        Returns:
            分割后的子图列表
        """
        if self._can_deploy_on_single_node(graph):
            return [graph]
        
        # 获取拓扑排序
        try:
            sorted_nodes = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            print("警告：图包含循环依赖")
            return [graph]
        
        if len(sorted_nodes) < 2:
            return [graph]
        
        # 二分切割
        cut_point = len(sorted_nodes) // 2
        va = sorted_nodes[:cut_point]
        vb = sorted_nodes[cut_point:]
        
        # 验证分割点
        if not va or not vb:
            if cut_point == 0:
                cut_point = 1
            elif cut_point == len(sorted_nodes):
                cut_point = len(sorted_nodes) - 1
            va = sorted_nodes[:cut_point]
            vb = sorted_nodes[cut_point:]
        
        # 递归分割
        segments = []
        if va:
            subgraph_a = graph.subgraph(va)
            segments.extend(self.split_tdg(subgraph_a))
        if vb:
            subgraph_b = graph.subgraph(vb)
            segments.extend(self.split_tdg(subgraph_b))
        
        return segments
    
    def _calculate_node_levels(self, graph: nx.DiGraph) -> Dict[int, int]:
        """计算DAG中每个节点的层级
        
        Args:
            graph: 要计算的图
            
        Returns:
            节点到层级的映射，层级从0开始
        """
        # 使用拓扑排序确保正确的层级计算顺序
        node_levels = {}
        for node in nx.topological_sort(graph):
            # 获取前驱节点的最大层级
            pred_levels = [node_levels.get(p, -1) for p in graph.predecessors(node)]
            # 当前节点的层级是前驱节点的最大层级+1
            node_levels[node] = max(pred_levels) + 1 if pred_levels else 0
        return node_levels

    def _can_deploy_on_single_node(self, graph: nx.DiGraph) -> bool:
        """检查子图是否可以部署在单个节点上
        
        Args:
            graph: 要检查的子图
            
        Returns:
            是否可以部署在单个节点上
        """
        # 计算每个节点的层级
        node_levels = self._calculate_node_levels(graph)
        max_level = max(node_levels.values()) if node_levels else 0
        
        # 检查最大层级是否超过最大阶段数
        # max_level从0开始，所以当max_level=min_stages-1时，正好使用min_stages个stage
        if max_level > self.min_stages - 1:
            return False
    
        # 检查每个VNF的资源需求
        # 获取图中每一层的节点
        levels = {}
        for node, level in node_levels.items():
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # 检查每一层的SRAM总和是否超过最小SRAM
        for level_nodes in levels.values():
            level_sram = sum(self._task.get_vnf_resource(node).sram for node in level_nodes)
            if level_sram > self.min_sram:
                return False
        return True

    def _deploy_segment(self, segment, switch, stage_offset=0):
        """将任务段部署到指定交换机
        
        Args:
            segment: 要部署的任务段
            switch: 目标交换机
            stage_offset: 阶段偏移量
        """
        # 计算每个节点的层级
        node_levels = self._calculate_node_levels(segment)
        
        # 按层级部署MAT到交换机的阶段
        for level in sorted(set(node_levels.values())):
            stage = stage_offset + level
            for node, node_level in node_levels.items():
                if node_level == level:
                    self._best_deployment.assign_vnf(node, switch, stage)

    def _select_switches(self, anchor_switch):
        """
        选择满足约束的候选交换机
        :param anchor_switch: 锚点交换机
        :return: 候选交换机列表
        """
        # 计算所有交换机到锚点交换机的延迟
        switch_latencies = []
        for switch in self._network.get_nodes():
            if switch == anchor_switch:
                continue
            path = self._network.get_shortest_path(anchor_switch, switch)
            latency = self._network.get_path_latency(path)
            if latency <= self.epsilon1:
                heapq.heappush(switch_latencies, (latency, switch))
        
        # 选择最近的ε2-1个交换机
        candidate_switches = [anchor_switch]
        for _ in range(min(self.epsilon2 - 1, len(switch_latencies))):
            _, switch = heapq.heappop(switch_latencies)
            candidate_switches.append(switch)
        
        return candidate_switches    

   

if __name__ == "__main__":
    import os
    import pandas as pd
    from netembd.network import Network
    from netembd.task import Task
    
    # 设置示例数据路径
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "examples")
    
    # 读取网络拓扑数据
    nodes_df = os.path.join(examples_dir, "FatTree_20", "nodes.csv")
    edges_df = os.path.join(examples_dir, "FatTree_20", "edges.csv")
    
    # 创建网络拓扑对象
    network = Network()
    network.load_nodes(nodes_df)
    network.load_edges(edges_df)
    # network.visualize()
    # 读取任务DAG数据
    mats_df = os.path.join(examples_dir, "merge_6", "mats.csv")
    deps_df = os.path.join(examples_dir, "merge_6", "deps.csv")
    
    # 创建任务DAG对象
    task = Task()
    task.load_vnfs(mats_df)
    task.load_dependencies(deps_df)
    
    # 创建优化器配置
    config = OptimizerConfig(time_limit=3600, gap_limit=0.01)
    
    # 创建并运行求解器
    try:
        optimizer = HermesHeuristic(network=network, task=task, config=config)
        deployment = optimizer.solve()
        if deployment:
            print("\n成功找到部署方案：")
            deployment.visualize()
            for vnf in task.get_vnfs():
                node = deployment.get_vnf_assignment(vnf).node_id
                stage = deployment.get_vnf_assignment(vnf).stage_id
                print(f"VNF {vnf} -> 节点 {node} -> stage {stage}")
            print(f"总延迟：{deployment.calculate_total_latency()}")
        else:
            print("\n未找到可行的部署方案")
            
    except Exception as e:
        print(f"\n求解过程出错：{str(e)}")
