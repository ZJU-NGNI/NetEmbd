"""数据收集路由优化模块

本模块实现了网络监控中的数据收集路由优化算法，主要包括：

1. MonPlanStage2: 使用 Gurobi ILP 的最优多路径选择
   - 基于整数线性规划(ILP)的精确求解
   - 支持多路径负载均衡
   - 考虑网络拥塞和链路容量约束
   - 最小化网络拥塞

算法特点：
- 精确求解：使用Gurobi求解器保证最优解
- 多路径支持：每个测量点可使用多条路径传输数据
- 拥塞感知：考虑链路容量和数据传输需求
- 可配置参数：支持调整路径数量等参数
"""

import time
import random
import heapq
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

from netembd.network import Network
from .MonPlan import Stage2Algorithm, Stage1Result, Stage2Result, MeasurementPoint, Flow

class BaseStage2Algorithm(Stage2Algorithm):
    """阶段二算法基类，提供通用功能"""
    
    def __init__(self, bandwidth_threshold: float = 0.8):
        """初始化基类
        
        Args:
            bandwidth_threshold: 带宽拥塞阈值
        """
        self.bandwidth_threshold = bandwidth_threshold

    def _get_control_nodes(self, network: Network) -> List[int]:
        """获取控制节点列表
        
        Args:
            network: 网络拓扑
            
        Returns:
            控制节点ID列表
        """
        control_nodes = []
        for node in network.get_nodes():
            resource = network.get_node_resource(node)
            if resource.control_node:
                control_nodes.append(node)
        return control_nodes
    
    def _estimate_data_size(self, measurement_point: MeasurementPoint, network: Network = None, large_flow_threshold: float = 100.0) -> float:
        """估算测量点的数据大小
        
        Args:
            measurement_point: 测量点
            network: 网络拓扑，用于获取flows列表
            large_flow_threshold: 大流阈值
            
        Returns:
            数据大小（MB）
        """
        total_size = 0.0
        covered_flows = measurement_point.covered_flows
        for measurement_type in measurement_point.measurement_types:
            if measurement_type == 'sketch':
                # sketch随机大小（1-1000MB）
                import random
                total_size += random.uniform(1.0, 1000.0)
            elif measurement_type == 'INT':
                # INT为测量流的百分比（默认1%）
                if covered_flows and network and hasattr(network, 'flows'):
                    # 找到 flow_id 为 covered_flows中的值 的流，得到流的大小，如果小于large_flow_threshold，则将大小加入flow_data_total
                    flow_data_total = 0.0
                    for flow in network.flows:
                        if flow.flow_id in covered_flows and flow.size < large_flow_threshold:
                            flow_data_total += flow.size
                    total_size += flow_data_total * 0.01  # 1%
                else:
                    # 如果没有流信息，使用默认估算
                    total_size += len(measurement_point.covered_flows) * 0.01
        
        return max(total_size, 0.1)  # 最小0.1MB
    
    def _precompute_paths_and_data(self, network: Network, stage1_result: Stage1Result) -> tuple:
        """预计算所有可能的路径和数据大小
        
        Returns:
            tuple: (all_paths, data_sizes) 其中all_paths是路径字典，data_sizes是数据大小字典
        """
        control_nodes = self._get_control_nodes(network)
        measurement_points = stage1_result.measurement_points
        all_paths = {}
        data_sizes = {}
        
        for measurement_point in measurement_points:
            measurement_node = measurement_point.node_id
            data_size = self._estimate_data_size(measurement_point, network)
            data_sizes[measurement_node] = data_size
            
            all_paths[measurement_node] = []
            
            for control_node in control_nodes:
                paths = self._find_multiple_paths(network, measurement_node, control_node)
                for path in paths:
                    if path:
                        all_paths[measurement_node].append(path)
        
        return all_paths, data_sizes

    def _find_multiple_paths(self, network: Network, source: int, target: int, max_paths: int = 3) -> List[List[int]]:
        """使用Yen's算法找到k条最短路径
        
        Args:
            network: 网络拓扑
            source: 源节点
            target: 目标节点
            max_paths: 最大路径数
            
        Returns:
            路径列表，按路径长度排序
        """
        return self._yen_k_shortest_paths(network, source, target, max_paths)
    
    def _yen_k_shortest_paths(self, network: Network, source: int, target: int, k: int) -> List[List[int]]:
        """Yen's k-shortest paths算法实现
        
        Args:
            network: 网络拓扑
            source: 源节点
            target: 目标节点
            k: 需要找到的路径数量
            
        Returns:
            k条最短路径列表
        """
        # 存储最终的k条最短路径
        A = []
        # 候选路径的优先队列 (路径长度, 路径)
        B = []
        
        # 找到最短路径
        shortest_path = self._dijkstra_path(network, source, target)
        if not shortest_path:
            return []
        
        A.append(shortest_path)
        
        for k_i in range(1, k):
            # 对于前一条路径中的每个节点
            prev_path = A[k_i - 1]
            
            for i in range(len(prev_path) - 1):
                # 分离路径
                root_path = prev_path[:i + 1]
                spur_node = prev_path[i]
                
                # 创建网络的副本用于修改
                removed_edges = set()
                
                # 移除与已找到路径共享根路径的路径的下一条边
                for path in A:
                    if len(path) > i and path[:i + 1] == root_path:
                        if i + 1 < len(path):
                            edge_to_remove = (path[i], path[i + 1])
                            removed_edges.add(edge_to_remove)
                
                # 移除根路径中的节点（除了spur_node）
                removed_nodes = set(root_path[:-1])
                
                # 在修改后的图中找到从spur_node到target的最短路径
                spur_path = self._dijkstra_path_with_restrictions(
                    network, spur_node, target, removed_edges, removed_nodes
                )
                
                if spur_path:
                    # 组合根路径和spur路径
                    total_path = root_path[:-1] + spur_path
                    path_length = self._calculate_path_length(network, total_path)
                    
                    # 检查路径是否已存在
                    if total_path not in [path for _, path in B] and total_path not in A:
                        heapq.heappush(B, (path_length, total_path))
            
            if not B:
                break
            
            # 选择最短的候选路径
            _, shortest_candidate = heapq.heappop(B)
            A.append(shortest_candidate)
        
        return A
    
    def _dijkstra_path(self, network: Network, source: int, target: int) -> List[int]:
        """使用Dijkstra算法找到最短路径
        
        Args:
            network: 网络拓扑
            source: 源节点
            target: 目标节点
            
        Returns:
            最短路径，如果不存在则返回None
        """
        return network.get_shortest_path(source, target)
    
    def _dijkstra_path_with_restrictions(self, network: Network, source: int, target: int, 
                                       removed_edges: Set[Tuple[int, int]], 
                                       removed_nodes: Set[int]) -> List[int]:
        """在有边和节点限制的情况下使用Dijkstra算法找到最短路径
        
        Args:
            network: 网络拓扑
            source: 源节点
            target: 目标节点
            removed_edges: 被移除的边集合
            removed_nodes: 被移除的节点集合
            
        Returns:
            最短路径，如果不存在则返回None
        """
        if source in removed_nodes or target in removed_nodes:
            return None
        
        # 初始化距离和前驱节点
        distances = {node: float('inf') for node in network.get_nodes()}
        predecessors = {}
        distances[source] = 0
        
        # 优先队列：(距离, 节点)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_node == target:
                break
            
            # 检查所有邻居
            for neighbor in network.get_neighbors(current_node):
                if neighbor in removed_nodes:
                    continue
                
                edge = (current_node, neighbor)
                if edge in removed_edges:
                    continue
                
                # 计算边的权重（使用延迟）
                edge_weight = network.get_edge_latency(current_node, neighbor)
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # 重构路径
        if target not in predecessors and target != source:
            return None
        
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors.get(current)
        
        path.reverse()
        
        if path[0] != source:
            return None
        
        return path
    
    def _calculate_path_length(self, network: Network, path: List[int]) -> float:
        """计算路径的总长度（延迟）
        
        Args:
            network: 网络拓扑
            path: 路径节点列表
            
        Returns:
            路径总长度
        """
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += network.get_edge_latency(path[i], path[i + 1])
        
        return total_length

    def _find_nearest_control_node(self, network: Network, measurement_node: int, control_nodes: List[int]) -> Tuple[int, List[int]]:
        """找到最近的控制节点和路径
        
        Args:
            network: 网络拓扑
            measurement_node: 测量节点
            control_nodes: 控制节点列表
            
        Returns:
            (最近控制节点ID, 路径)
        """
        best_control = None
        best_path = None
        min_latency = float('inf')
        
        for control_node in control_nodes:
            path = network.get_shortest_path(measurement_node, control_node)
            if path is not None:
                latency = network.get_path_latency(path)
                if latency < min_latency:
                    min_latency = latency
                    best_control = control_node
                    best_path = path
        
        return best_control, best_path
    
    def _calculate_path_congestion(self, network: Network, data_paths: Dict[int, List[List[int]]], 
                                  data_sizes: Dict[int, float], flows: List = None) -> float:
        """计算路径拥塞率
        
        Args:
            network: 网络拓扑
            data_paths: 数据路径字典, 格式{测量节点ID: [路径1, 路径2, ...]}
            data_sizes: 数据大小字典, 格式{测量节点ID: 数据大小}
            flows: 背景流量列表
            
        Returns:
            拥塞率
        """
        # 从flow generator获取使用情况
        edge_capacity = network.flow_generator.available_bandwidth
        edge_usage = defaultdict(float)
        # 添加控制数据流
        for measurement_node, paths in data_paths.items():
            data_size = data_sizes.get(measurement_node, 0)
            
            for path in paths:
                # 如果有多条路径，数据平均分配
                path_data = data_size / len(paths)
                
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    edge_usage[edge] += path_data
        
        # 计算拥塞率：背景流+控制数据流>带宽的链路数量/总链路数量
        congested_edges = 0
        total_edges = len(edge_capacity)

        for edge, usage in edge_usage.items():
            capacity = edge_capacity[edge]
            if usage > capacity:  # 拥塞定义：总流量超过带宽
                congested_edges += 1
        
        return congested_edges / total_edges if total_edges > 0 else 0.0
    
    def _calculate_total_hops(self, data_paths: Dict[int, List[List[int]]]) -> int:
        """计算总跳数
        
        Args:
            data_paths: 数据路径字典
            
        Returns:
            总跳数
        """
        total_hops = 0
        for paths in data_paths.values():
            for path in paths:
                total_hops += len(path) - 1  # 跳数 = 路径长度 - 1
        return total_hops
    
    def _calculate_data_loss_rate(self, network: Network, data_paths: Dict[int, List[List[int]]], 
                                 data_sizes: Dict[int, float], flows: List = None) -> float:
        """计算数据丢失率
        
        Args:
            network: 网络拓扑
            data_paths: 数据路径字典, 格式{测量节点ID: [路径1, 路径2, ...]}
            data_sizes: 数据大小字典, 格式{测量节点ID: 数据大小}
            flows: 背景流量列表
            
        Returns:
            数据丢失率
        """
            
        # 从flow generator获取使用情况
        edge_capacity = network.flow_generator.available_bandwidth
        edge_usage = defaultdict(float)
        # 添加控制数据流
        for measurement_node, paths in data_paths.items():
            data_size = data_sizes.get(measurement_node, 0)
            
            for path in paths:
                # 如果有多条路径，数据平均分配
                path_data = data_size / len(paths)
                
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    edge_usage[edge] += path_data

        # 计算控制数据的丢失量
        # 对于每条控制数据路径，检查其是否能够成功传输
        total_lost_control_data = 0.0
        total_control_data = 0.0
        for edge, usage in edge_usage.items():
            capacity = edge_capacity[edge]
            total_control_data += usage
            if usage > capacity:  # 拥塞定义：总流量超过带宽
                total_lost_control_data += usage - capacity
                    
        return total_lost_control_data / total_control_data if total_control_data > 0 else 0.0

class MonPlanStage2(BaseStage2Algorithm):
    """MonPlan阶段二算法：使用Gurobi ILP实现最优化的多路径选择
    
    该算法通过整数线性规划(ILP)求解最优的数据收集路径，
    主要目标是最小化网络拥塞，同时保证所有测量点的数据能够传输到控制节点。
    """
    
    def __init__(self, max_paths_per_node: int = 3, time_limit: int = 300, **kwargs):
        """初始化MonPlan算法
        
        Args:
            max_paths_per_node: 每个测量点的最大路径数
            time_limit: Gurobi求解时间限制（秒）
        """
        super().__init__(**kwargs)
        self.max_paths_per_node = max_paths_per_node
        self.time_limit = time_limit
    
    def solve(self, network: Network, stage1_result: Stage1Result, flows: List[Flow] = None) -> Stage2Result:
        """求解MonPlan数据收集问题
        
        Args:
            network: 网络拓扑
            stage1_result: 阶段一结果
            flows: 背景流量列表
            
        Returns:
            Stage2Result: 包含最优路径和评估指标的结果
        """
        start_time = time.time()
        
        # 预计算所有可能的路径和数据大小
        all_paths, data_sizes = self._precompute_paths_and_data(network, stage1_result)
        
        # 获取控制节点列表
        control_nodes = self._get_control_nodes(network)
        
        # 创建Gurobi模型
        model = gp.Model('monplan_stage2')
        
        # 决策变量：x[m][p] 表示测量点m是否使用路径p
        x = {}
        for m in all_paths:
            x[m] = {}
            for p_idx, path in enumerate(all_paths[m]):
                x[m][p_idx] = model.addVar(vtype=GRB.BINARY, name=f'x_{m}_{p_idx}')
        
        # 约束1：每个测量点至少选择一条路径
        for m in all_paths:
            model.addConstr(gp.quicksum(x[m][p_idx] for p_idx in range(len(all_paths[m]))) >= 1)
        
        # 约束2：链路容量约束
        edge_capacity = network.flow_generator.available_bandwidth
        edge_usage = defaultdict(gp.LinExpr)
        for m in all_paths:
            data_size = data_sizes[m]
            for p_idx, path in enumerate(all_paths[m]):
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    edge_usage[edge] += data_size * x[m][p_idx] / len(all_paths[m])
        
        # 引入辅助变量y表示最大链路利用率
        y = model.addVar(name='max_utilization')
        for edge, usage in edge_usage.items():
            capacity = edge_capacity[edge]
            model.addConstr(usage <= y * capacity)
        
        # 目标函数：最小化最大链路利用率
        model.setObjective(y, GRB.MINIMIZE)
        
        # 求解模型
        model.optimize()
        
        # 提取结果
        data_paths = {}
        for m in all_paths:
            data_paths[m] = []
            for p_idx, path in enumerate(all_paths[m]):
                if x[m][p_idx].X > 0.5:  # 二进制变量，>0.5表示选中
                    data_paths[m].append(path)
        
        # 计算评估指标
        total_hops = self._calculate_total_hops(data_paths)
        congestion_rate = self._calculate_path_congestion(network, data_paths, data_sizes, flows)
        data_loss_rate = self._calculate_data_loss_rate(network, data_paths, data_sizes, flows)
        execution_time = time.time() - start_time
        
        return Stage2Result(
            data_paths=data_paths,
            total_hop_count=total_hops,
            congestion_rate=congestion_rate,
            data_loss_rate=data_loss_rate,
            execution_time=execution_time
        )

class IntPathStage2(BaseStage2Algorithm):
    """IntPath阶段二算法：使用DFS计算到 nearest control node的一条路径"""
    
    def __init__(self, **kwargs):
        """初始化IntPath算法"""
        super().__init__(**kwargs)
    
    def _dfs_path(self, network: Network, source: int, target: int) -> List[int]:
        """使用DFS算法找到从源节点到目标节点的一条路径
        
        Args:
            network: 网络拓扑
            source: 源节点
            target: 目标节点
            
        Returns:
            List[int]: 找到的路径，如果没有找到则返回None
        """
        visited = set()
        path = []
        
        def dfs(current: int) -> bool:
            if current == target:
                return True
            
            visited.add(current)
            path.append(current)
            
            # 获取当前节点的所有邻居节点
            neighbors = network.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            
            path.pop()
            return False
        
        if dfs(source):
            path.append(target)
            return path
        return None
    
    def solve(self, network: Network, stage1_result: Stage1Result, flows: List[Flow] = None) -> Stage2Result:
        """求解IntPath数据收集问题
        
        Args:
            network: 网络拓扑
            stage1_result: 阶段一结果
            flows: 背景流量列表
            
        Returns:
            Stage2Result: 包含最优路径和评估指标的结果
        """
        start_time = time.time()
        
        control_nodes = self._get_control_nodes(network)
        data_paths = {}
        data_sizes = {}
        
        # 为每个测量点找到到最近控制节点的路径
        for measurement_point in stage1_result.measurement_points:
            measurement_node = measurement_point.node_id
            data_size = self._estimate_data_size(measurement_point, network)
            data_sizes[measurement_node] = data_size
            
            # 找到最近的控制节点和路径
            control_node = self._find_nearest_control_node(network, measurement_node, control_nodes)[0]
            
            # 使用DFS算法找到从measurement_point到control_node的路径
            path = self._dfs_path(network, measurement_node, control_node)
            
            if path:
                data_paths[measurement_node] = [path]
            else:
                # 如果没有找到路径，使用空列表
                data_paths[measurement_node] = []
        
        # 计算评估指标
        total_hops = self._calculate_total_hops(data_paths)
        congestion_rate = self._calculate_path_congestion(network, data_paths, data_sizes, flows)
        data_loss_rate = self._calculate_data_loss_rate(network, data_paths, data_sizes, flows)
        execution_time = time.time() - start_time
        
        return Stage2Result(
            data_paths=data_paths,
            total_hop_count=total_hops,
            congestion_rate=congestion_rate,
            data_loss_rate=data_loss_rate,
            execution_time=execution_time
        )
    
class EscalaStage2(BaseStage2Algorithm):
    """Escala阶段二算法：低延迟路径选择"""
    
    def __init__(self, max_paths_per_node: int = 3, **kwargs):
        """初始化Escala算法
        
        Args:
            max_paths_per_node: 每个测量点的最大路径数
        """
        super().__init__(**kwargs)
        self.max_paths_per_node = max_paths_per_node
    
    def solve(self, network: Network, stage1_result: Stage1Result, flows: List[Flow] = None) -> Stage2Result:
        """求解Escala数据收集问题"""
        start_time = time.time()
        
        control_nodes = self._get_control_nodes(network)
        data_paths = {}
        data_sizes = {}
        
        # 为每个测量点选择最优路径
        for measurement_point in stage1_result.measurement_points:
            measurement_node = measurement_point.node_id
            data_size = self._estimate_data_size(measurement_point, network)
            data_sizes[measurement_node] = data_size
            
            
            control_node, path = self._find_nearest_control_node(network, measurement_node, control_nodes)

            selected_paths = [path, ]

            data_paths[measurement_node] = selected_paths
        
        # 计算评估指标
        total_hops = self._calculate_total_hops(data_paths)
        congestion_rate = self._calculate_path_congestion(network, data_paths, data_sizes, flows)
        data_loss_rate = self._calculate_data_loss_rate(network, data_paths, data_sizes, flows)
        execution_time = time.time() - start_time
        
        return Stage2Result(
            data_paths=data_paths,
            total_hop_count=total_hops,
            congestion_rate=congestion_rate,
            data_loss_rate=data_loss_rate,
            execution_time=execution_time
        )

class RandomStage2(BaseStage2Algorithm):
    """Random阶段二算法：随机路径选择"""
    
    def __init__(self, max_paths_per_node: int = 2, random_seed: int = 42, **kwargs):
        """初始化Random算法
        
        Args:
            max_paths_per_node: 每个测量点的最大路径数
            random_seed: 随机种子
        """
        super().__init__(**kwargs)
        self.max_paths_per_node = max_paths_per_node
        self.random_seed = random_seed
    
    def solve(self, network: Network, stage1_result: Stage1Result, flows: List[Flow] = None) -> Stage2Result:
        """求解Random数据收集问题"""
        start_time = time.time()
        random.seed(self.random_seed)
        
        control_nodes = self._get_control_nodes(network)
        data_paths = {}
        data_sizes = {}
        
        # 为每个测量点随机选择路径
        for measurement_point in stage1_result.measurement_points:
            measurement_node = measurement_point.node_id
            data_size = self._estimate_data_size(measurement_point, network)
            data_sizes[measurement_node] = data_size
            
            control_node = self._find_nearest_control_node(network, measurement_node, control_nodes)[0]

            # 尝试获取所有可能的路径
            candidate_paths = self._find_multiple_paths(network, measurement_node, control_node, max_paths=10)
            if candidate_paths:
                # 从候选路径中随机选择一条
                path = random.choice(candidate_paths)
                selected_paths = [path]
            else:
                # 如果没有找到路径，尝试使用最短路径作为备选
                shortest_path = network.get_shortest_path(measurement_node, control_node)
                selected_paths = [shortest_path] if shortest_path else []
            
            data_paths[measurement_node] = selected_paths
        
        # 计算评估指标
        total_hops = self._calculate_total_hops(data_paths)
        congestion_rate = self._calculate_path_congestion(network, data_paths, data_sizes, flows)
        data_loss_rate = self._calculate_data_loss_rate(network, data_paths, data_sizes, flows)
        execution_time = time.time() - start_time
        
        return Stage2Result(
            data_paths=data_paths,
            total_hop_count=total_hops,
            congestion_rate=congestion_rate,
            data_loss_rate=data_loss_rate,
            execution_time=execution_time
        )