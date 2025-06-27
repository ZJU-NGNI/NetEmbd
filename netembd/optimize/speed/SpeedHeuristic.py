from typing import Dict, List, Optional, Set, Tuple
from netembd.interfaces import OptimizerConfig, OptimizationStatus
from netembd.network import Network
from netembd.task import Task
from netembd.optimize import HeuristicOptimizer
from netembd.deployment import Deployment
from netembd.interfaces.base_network import NodeResource

class SpeedHeuristic(HeuristicOptimizer):
    """Speed启发式算法实现类"""
    
    def __init__(self, network: Network, task: Task, config: Optional[OptimizerConfig] = None):
        """初始化Speed启发式算法
        
        Args:
            network: 网络拓扑对象
            task: 任务DAG对象
            config: 优化器配置对象
        """
        super().__init__(network, task, config)
        
    def _stage_count_per_switch_min(self) -> int:
        """计算每个交换机上的stage数量最小值
        
        Returns:
            每个交换机上的stage数量最小值
        """
        return min(self._network.get_node_resource(node).stage for node in self._network.get_nodes())

    def _sram_per_stage_min(self) -> int:
        """计算每个stage的SRAM资源最小值
        
        Returns:
            每个stage的SRAM资源最小值
        """
        return min(self._network.get_node_resource(node).sram for node in self._network.get_nodes())

    def _get_obs_id(self, stage_id:int, stage_per_switch:int) -> int:
        """根据stage ID获取其对应的OBS交换机ID
        
        Args:
            stage_id: stage ID
            stage_per_switch: 每个交换机上的stage数量
        
        Returns:
            对应的OBS交换机ID
        """
        return stage_id // stage_per_switch

    def _get_vnf_resource(self, mat_id:int) -> NodeResource:
        """根据VNF ID获取其对应的资源需求
        
        Args:
            mat_id: VNF ID
        
        Returns:
            对应的资源需求对象
        """
        return self._task.get_vnf_resource(mat_id)
            
    def _calculate_mat_levels(self) -> Dict[int, int]:
        """计算每个VNF的层级
        
        Returns:
            每个VNF的层级列表
        """
        vnfs = self._task.get_vnfs()
        mat_levels = {vnf: 0 for vnf in vnfs}
        mat_degrees = {vnf: 0 for vnf in vnfs}
        
        # 计算入度
        for src, dst in self._task.get_dependencies():
            mat_degrees[dst] += 1
            
        def calculate_level(mat: int):
            if mat_levels[mat] != 0:
                return
            sub_levels = []
            for i, j in self._task.get_dependencies():
                if i == mat:
                    calculate_level(j)
                    sub_levels.append(mat_levels[j])
            mat_levels[mat] = max(sub_levels) + 1 if sub_levels else 1
            
        # 从入度为0的节点开始计算层级
        for mat, degree in mat_degrees.items():
            if degree == 0:
                calculate_level(mat)
                
        return mat_levels
        
    def _place_on_obs_ffl(self) -> Dict[int, List[int]]:
        """使用FFL算法将任务部署到OBS上
        
        Returns:
            stage到VNF的映射字典，key为stage ID，value为该stage上部署的VNF列表
        """
        # 获取VNF数量和stage数量
        mats = self._task.get_vnfs()
        switches = self._network.get_nodes()
        stage_per_switch = self._stage_count_per_switch_min()
        sram_per_stage = self._sram_per_stage_min()
        stage_count = len(switches) * stage_per_switch
        
        # 计算每个VNF的层级
        mat_levels = self._calculate_mat_levels()
        
        # 按层级从高到低排序VNF
        mat_order = sorted(mats, key=lambda x: mat_levels[x], reverse=True)
        
        # 初始化stage资源
        stage_resources = [sram_per_stage for _ in range(stage_count)]
        
        # 初始化映射结果
        stage_to_mats = {}
        mat_to_stage = {vnf: -1 for vnf in mats}
        
        # 记录已使用的stage
        used_stages = set()
        max_used_stage = -1
        
        # 从高层级到低层级依次部署VNF
        for mat in mat_order:
            # 获取VNF的资源需求
            mat_resource = self._get_vnf_resource(mat).sram
            
            # 检查依赖关系
            min_stage = 0
            for src, dst in self._task.get_dependencies():
                if dst == mat and src in mat_to_stage:
                    min_stage = max(min_stage, mat_to_stage[src] + 1)
            
            # 找到第一个满足条件的stage
            assigned = False
            for stage in range(min_stage, stage_count):
                # 检查资源约束
                if stage_resources[stage] < mat_resource:
                    continue
                    
                # 检查stage连续性
                if stage > max_used_stage + 1:
                    continue
                    
                # 更新stage资源和映射
                stage_resources[stage] -= mat_resource
                if stage not in stage_to_mats:
                    stage_to_mats[stage] = []
                stage_to_mats[stage].append(mat)
                mat_to_stage[mat] = stage
                used_stages.add(stage)
                max_used_stage = max(max_used_stage, stage)
                assigned = True
                break
                
            if not assigned:
                self._status = OptimizationStatus.ERROR
                raise RuntimeError("无法找到满足依赖关系和资源约束的部署方案")
                
        return stage_to_mats
        
    def _place_on_obs_ffls(self) -> Dict[int, List[int]]:
        """使用FFLS算法将任务部署到OBS上
        
        Returns:
            stage到VNF的映射字典，key为stage ID，value为该stage上部署的VNF列表
        """
        # 获取VNF数量和stage数量
        mats = self._task.get_vnfs()
        switches = self._network.get_nodes()
        stage_per_switch = self._stage_count_per_switch_min()
        sram_per_stage = self._sram_per_stage_min()
        stage_count = len(switches) * stage_per_switch
        
        # 计算每个VNF的层级
        mat_levels = self._calculate_mat_levels()
        
        # 按层级从高到低排序VNF
        mat_order = sorted(mats, key=lambda x: mat_levels[x], reverse=True)
        
        # 初始化stage资源
        stage_resources = [sram_per_stage for _ in range(stage_count)]
        
        # 初始化映射结果
        stage_to_mats = {}
        mat_to_stage = {vnf: -1 for vnf in mats}
        
        # 记录已使用的stage
        used_stages = set()
        max_used_stage = -1
        
        # 从高层级到低层级依次部署VNF
        for mat in mat_order:
            # 获取VNF的资源需求
            mat_resource = self._get_vnf_resource(mat).sram
            
            # 检查依赖关系
            min_stage = 0
            for src, dst in self._task.get_dependencies():
                if dst == mat and src in mat_to_stage:
                    min_stage = max(min_stage, mat_to_stage[src] + 1)
            
            # 找到资源最少的合法stage
            min_resource = float('inf')
            best_stage = -1
            for stage in range(min_stage, stage_count):
                # 检查资源约束
                if stage_resources[stage] < mat_resource:
                    continue
                    
                # 检查stage连续性
                if stage > max_used_stage + 1:
                    continue
                    
                # 更新最优stage
                if stage_resources[stage] < min_resource:
                    min_resource = stage_resources[stage]
                    best_stage = stage
            
            if best_stage == -1:
                self._status = OptimizationStatus.ERROR
                raise RuntimeError("无法找到满足依赖关系和资源约束的部署方案")
            
            # 更新stage资源和映射
            stage_resources[best_stage] -= mat_resource
            if best_stage not in stage_to_mats:
                stage_to_mats[best_stage] = []
            stage_to_mats[best_stage].append(mat)
            mat_to_stage[mat] = best_stage
            used_stages.add(best_stage)
            max_used_stage = max(max_used_stage, best_stage)
                
        return stage_to_mats
        
    def _place_on_network_r_greedy(self, stage_to_mats: Dict[int, List[int]]) -> Dict[int, int]:
        """使用R-Greedy算法将OBS部署到网络拓扑上
        
        Args:
            stage_to_mats: stage到VNF的映射字典
            
        Returns:
            OBS交换机到物理交换机的映射字典
        """
        # 计算需要的交换机数量
        stage_per_switch = self._stage_count_per_switch_min()
        obs_stage_count = len(stage_to_mats)
        obs_switch_count = (obs_stage_count + stage_per_switch - 1) // stage_per_switch
        
        # 计算每个节点的资源总量和可用stage数
        node_resources = {}
        node_stages = {}
        for node in self._network.get_nodes():
            resource = self._network.get_node_resource(node)
            node_resources[node] = resource.stage * resource.sram
            node_stages[node] = resource.stage
        
        # 按资源总量从大到小排序节点
        sorted_nodes = sorted(self._network.get_nodes(), 
                            key=lambda x: (node_resources[x], node_stages[x]), 
                            reverse=True)
        
        # 选择资源最多且stage数足够的节点作为OBS交换机
        selected_switches = []
        for node in sorted_nodes:
            if len(selected_switches) >= obs_switch_count:
                break
            if node_stages[node] >= (obs_stage_count + len(selected_switches)) // obs_switch_count:
                selected_switches.append(node)
        
        if len(selected_switches) < obs_switch_count:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError("无法找到足够的物理节点来部署OBS")
        
        # 构建OBS交换机到物理交换机的映射
        obs_to_phy = {}
        for obs_id in range(obs_switch_count):
            obs_to_phy[obs_id] = selected_switches[obs_id]
                
        return obs_to_phy
        
    def _place_on_network_r_bfs(self, stage_to_mats: Dict[int, List[int]]) -> Dict[int, int]:
        """使用R-BFS算法将OBS部署到网络拓扑上
        
        Args:
            stage_to_mats: stage到VNF的映射字典
            
        Returns:
            OBS交换机到物理交换机的映射字典
        """
        # 计算需要的交换机数量
        stage_per_switch = self._stage_count_per_switch_min()
        obs_stage_count = len(stage_to_mats)
        obs_switch_count = (obs_stage_count + stage_per_switch - 1) // stage_per_switch
        
        # 计算每个节点的资源总量和可用stage数
        node_resources = {}
        node_stages = {}
        for node in self._network.get_nodes():
            resource = self._network.get_node_resource(node)
            node_resources[node] = resource.stage * resource.sram
            node_stages[node] = resource.stage
        
        # 按资源总量从大到小排序节点
        sorted_nodes = sorted(self._network.get_nodes(), 
                            key=lambda x: (node_resources[x], node_stages[x]), 
                            reverse=True)
        
        # 选择资源最多的节点作为起始点
        start_node = sorted_nodes[0]
        selected_switches = [start_node]
        available_nodes = set(self._network.get_nodes()) - {start_node}
        
        # 使用BFS选择剩余交换机
        while len(selected_switches) < obs_switch_count:
            # 初始化队列
            queue = []
            visited = set()
            
            # 将已选交换机的邻居加入队列
            for switch in selected_switches:
                for neighbor in self._network.get_neighbors(switch):
                    if neighbor in available_nodes and neighbor not in visited:
                        # 检查节点是否有足够的stage资源
                        required_stages = (obs_stage_count + len(selected_switches)) // obs_switch_count
                        if node_stages[neighbor] >= required_stages:
                            queue.append((neighbor, node_resources[neighbor]))
                            visited.add(neighbor)
            
            # 如果队列为空，说明无法找到更多合适的交换机
            if not queue:
                self._status = OptimizationStatus.ERROR
                raise RuntimeError("无法找到足够的物理节点来部署OBS")
            
            # 选择队列中资源最多的交换机
            best_node = max(queue, key=lambda x: x[1])[0]
            selected_switches.append(best_node)
            available_nodes.remove(best_node)
        
        # 构建OBS交换机到物理交换机的映射
        obs_to_phy = {}
        for obs_id in range(obs_switch_count):
            obs_to_phy[obs_id] = selected_switches[obs_id]
                
        return obs_to_phy
        
    def _place_on_network_noderank(self, stage_to_mats: Dict[int, List[int]]) -> Dict[int, int]:
        """使用NodeRank算法将OBS部署到网络拓扑上
        
        Args:
            stage_to_mats: stage到VNF的映射字典
            
        Returns:
            OBS交换机到物理交换机的映射字典
        """
        # 计算需要的交换机数量
        stage_per_switch = self._stage_count_per_switch_min()
        obs_stage_count = len(stage_to_mats)
        obs_switch_count = (obs_stage_count + stage_per_switch - 1) // stage_per_switch
        
        # 计算每个节点的权重（连接数）
        node_weights = {}
        for node in self._network.get_nodes():
            weight = 0
            for src, dst in self._network.get_edges():
                if src == node or dst == node:
                    weight += 1
            node_weights[node] = weight
            
        # 按权重从高到低排序节点
        sorted_nodes = sorted(node_weights.keys(), key=lambda x: node_weights[x], reverse=True)
        
        # 选择权重最高的节点作为根节点
        root = sorted_nodes[0]
        selected_switches = [root]
        sorted_nodes.remove(root)
        
        # 使用BFS选择剩余交换机
        queue = [root]
        visited = {root}
        
        while len(selected_switches) < obs_switch_count and queue:
            current = queue.pop(0)
            
            # 获取相邻节点
            neighbors = set()
            for src, dst in self._network.get_edges():
                if src == current and dst in sorted_nodes:
                    neighbors.add(dst)
                elif dst == current and src in sorted_nodes:
                    neighbors.add(src)
                    
            # 将未访问的相邻节点按权重排序
            neighbors = sorted(neighbors, key=lambda x: node_weights[x], reverse=True)
            
            # 将未访问的相邻节点加入队列
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    selected_switches.append(neighbor)
                    sorted_nodes.remove(neighbor)
                    if len(selected_switches) == obs_switch_count:
                        break
                        
        if len(selected_switches) < obs_switch_count:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError("网络连通性不足，无法完成部署")
            
        # 构建OBS交换机到物理交换机的映射
        obs_to_phy = {}
        for obs_id in range(obs_switch_count):
            obs_to_phy[obs_id] = selected_switches[obs_id]
                
        return obs_to_phy
        
    def solve(self) -> Optional[Deployment]:
        """求解部署优化问题
        
        Returns:
            最优部署方案，如果未找到可行解则返回None
        """
        try:
            # 第一步：将任务部署到OBS上
            # 可以选择FFL或FFLS算法
            stage_to_mats = self._place_on_obs_ffl()
            # stage_to_mats = self._place_on_obs_ffls()
            
            # 第二步：将OBS部署到网络拓扑上
            # 可以选择R-Greedy、R-BFS或NodeRank算法
            obs_to_phy = self._place_on_network_r_greedy(stage_to_mats)
            # obs_to_phy = self._place_on_network_r_bfs(stage_to_mats)
            # obs_to_phy = self._place_on_network_noderank(stage_to_mats)
            
            # 创建部署方案
            deployment = Deployment(self._network, self._task)
            stage_per_switch = self._stage_count_per_switch_min()
            
            # 遍历obs_to_phy，将每个obs交换机映射到对应的物理交换机
            for obs_switch, phy_switch in obs_to_phy.items():
                # 计算该obs交换机的第一个stage编号
                first_stage = obs_switch * stage_per_switch
                # 遍历该obs交换机的所有stage
                for stage_id in range(first_stage, first_stage + stage_per_switch):
                    # 获取该stage的所有vnfs，如果stage不存在则跳过
                    if stage_id not in stage_to_mats:
                        continue
                    vnfs = stage_to_mats[stage_id]
                    # 遍历该stage的所有vnfs
                    for vnf_id in vnfs:
                        # 将vnf分配到对应的物理交换机
                        deployment.assign_vnf(vnf_id, phy_switch, stage_id%stage_per_switch)
            
            # 保存部署方案到类成员变量
            self._best_deployment = deployment
            self._best_objective = deployment.calculate_total_latency()
            self._status = OptimizationStatus.OPTIMAL
            
            return deployment
            
        except Exception as e:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError(f"求解过程出错：{str(e)}")

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
        optimizer = SpeedHeuristic(network, task, config)
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

