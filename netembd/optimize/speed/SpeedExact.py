"""Speed精确求解器实现模块。

此模块实现了基于Speed策略的精确求解器，继承自ExactOptimizer基类。
求解过程分为两步：
1. 将任务部署到OBS (Optimized Binary Search)上
2. 将OBS部署到网络拓扑上

Typical usage example:

    from netembd.optimize.Speed import SpeedExact
    
    optimizer = SpeedExact(network, task, OptimizerConfig(time_limit=3600))
    deployment = optimizer.solve()
"""

from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from netembd.deployment import Deployment
from netembd.network import Network
from netembd.interfaces.base_network import NodeResource
from netembd.task import Task
from netembd.optimize import ExactOptimizer
from netembd.interfaces.base_optimizer import OptimizerConfig, OptimizationStatus

class SpeedExact(ExactOptimizer):
    """Speed精确求解器实现类"""
    
    def __init__(self, network: Network, task: Task, config: Optional[OptimizerConfig] = None):
        """初始化Speed精确求解器
        
        Args:
            network: 网络拓扑对象
            task: 任务DAG对象
            config: 优化器配置对象
        """
        super().__init__(network, task, config)

        
    def _calculate_mat_levels(self) -> List[int]:
        """计算每个VNF的层级
        
        Returns:
            每个VNF的层级列表
        """
        mat_levels = [0] * len(self._task.get_vnfs())
        mat_degrees = [0] * len(self._task.get_vnfs())
        
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
        for mat, degree in enumerate(mat_degrees):
            if degree == 0:
                calculate_level(mat)
                
        return mat_levels
        
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

    def _place_on_obs(self) -> Dict[int, List[int]]:
        """将任务部署到OBS上
        
        Returns:
            stage到VNF的映射字典，key为stage ID，value为该stage上部署的VNF列表
        """
        model = gp.Model("Place_OBS")
        
        # 获取VNF数量和stage数量
        swtiches = self._network.get_nodes()
        mats = self._task.get_vnfs()

        mat_count = len(self._task.get_vnfs())
        swtich_count = len(self._network.get_nodes())
        stage_per_switch = self._stage_count_per_switch_min()   # 使用每个交换机的最小stage数量，在obs部署时更方便
        sram_per_stage = self._sram_per_stage_min()   # 每个stage的SRAM资源限制
        stage_count = swtich_count * stage_per_switch

        stages = [stage for stage in range(stage_count)]

        # 计算每个VNF的层级
        # mat_levels = self._calculate_mat_levels()
        
        # 定义决策变量
        offload_vnfId_to_stageId = model.addVars(mats, stages, vtype=GRB.BINARY, name='offload_vnfId_to_stageId')  # x[i,j]=1表示VNF i部署到stage j
        stageUsedFlag = model.addVars(stages, vtype=GRB.BINARY, name='stageUsedFlag')  # y[j]=1表示stage j被使用
        stageId_for_vnfId = model.addVars(mats, vtype=GRB.INTEGER, name='stageId_for_vnfId')  # z[i]表示VNF i部署的stage编号
        
        # x和y之间的逻辑约束
        for stage in stages:
            model.addGenConstrOr(stageUsedFlag[stage], [offload_vnfId_to_stageId[mat,stage] for mat in mats], "orconstr")
            
        # 如果VNF i部署到stage j，则z[i]=j
        for mat in mats:
            for stage in stages:
                model.addConstr((offload_vnfId_to_stageId[mat,stage] == 1) >> (stageId_for_vnfId[mat] == stage))
                
        # 满足依赖关系：如果VNF i依赖VNF j，则j必须在i之前的stage
        for i, j in self._task.get_dependencies():
            model.addConstr(stageId_for_vnfId[j] >= stageId_for_vnfId[i] + 1)
            
        # 每个stage的资源使用不超过容量
        for stage in stages:
            model.addConstr(
                gp.quicksum(offload_vnfId_to_stageId[mat,stage] * self._get_vnf_resource(mat).sram 
                          for mat in mats) <= sram_per_stage,
                f"sram_limit_{stage}"
            )
            
        # 每个VNF必须部署到一个stage
        for i in mats:
            model.addConstr(gp.quicksum(offload_vnfId_to_stageId[i,j] for j in stages) == 1,
            f"must_map_{i}")
            
        # stage必须连续使用
        for j in stages:
            if j > 0:
                model.addConstr(stageUsedFlag[j] >= stageUsedFlag[j-1],
                f"consecutive_stage_{j}")
            
        # 目标：最小化使用的stage数量
        model.setObjective(gp.quicksum(stageUsedFlag[j] for j in stages), GRB.MINIMIZE)
        
        # 设置求解参数
        if self._config:
            model.setParam('TimeLimit', self._config.time_limit)
            model.setParam('MIPGap', self._config.gap_limit)
            model.setParam('OutputFlag', self._config.verbose)
            
        # 求解
        model.optimize()
        
        if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError(f"OBS部署优化求解失败")
            
        # 收集结果
        stage_to_mats = {}
        for i in mats:
            for j in stages:
                if offload_vnfId_to_stageId[i,j].x > 0.5:
                    if j not in stage_to_mats:
                        stage_to_mats[j] = []
                    stage_to_mats[j].append(i)
                    
        # 由于已经有consecutive约束保证了stage连续使用
        # 将stage_to_mats remap到从0号OBS开始
        # print("原始stage_to_mats:", stage_to_mats)
        remapped_stage_to_mats = {}
        sorted_stages = sorted(stage_to_mats.keys())
        for new_stage_id, old_stage_id in enumerate(sorted_stages):
            remapped_stage_to_mats[new_stage_id] = stage_to_mats[old_stage_id]
        # print("重映射后stage_to_mats:", remapped_stage_to_mats)
        return remapped_stage_to_mats
                
    def _place_on_network(self, stage_to_mats: Dict[int, List[int]]) -> Dict[int, int]:
        """将OBS部署到网络拓扑上
        
        Args:
            stage_to_mats: stage到VNF的映射字典
            
        Returns:
            obs交换机到物理交换机的映射字典
        """
        model = gp.Model("Place_Network")
        stage_per_switch = self._stage_count_per_switch_min()
        # 计算需要的交换机数量
        obs_stage_count = len(stage_to_mats)
        obs_switch_count = (obs_stage_count + stage_per_switch - 1) // stage_per_switch
        
        # 获取物理节点数量
        phy_switch_count = len(self._network.get_nodes())
        
        # 定义决策变量
        map_obsId_to_phyId = model.addVars(obs_switch_count, phy_switch_count, vtype=GRB.BINARY, name='map_obsId_to_phyId')  # a[i,j]=1表示OBS交换机i映射到物理交换机j
        phyId_for_obsId = model.addVars(obs_switch_count, vtype=GRB.INTEGER, name='phyId_for_obsId')  # 存储OBS节点映射结果
        latency_for_obsEdge = model.addVars(obs_switch_count-1, vtype=GRB.INTEGER, name='latency_for_obsEdge')  # OBS边实际的时延
        obsEdge_to_phyEdge = model.addVars(obs_switch_count-1, phy_switch_count, phy_switch_count, vtype=GRB.BINARY, name='obsEdge_to_phyEdge')  # OBS边映射到物理边
        
        # 每个OBS交换机必须映射到一个物理交换机
        for i in range(obs_switch_count):
            model.addConstr(gp.quicksum(map_obsId_to_phyId[i,j] for j in range(phy_switch_count)) == 1,
            f"must_map_obs_{i}")
            
        # 每个物理交换机最多被一个OBS交换机使用
        for j in range(phy_switch_count):
            model.addConstr(gp.quicksum(map_obsId_to_phyId[i,j] for i in range(obs_switch_count)) <= 1,
            f"at_most_map_phy_{j}")
            
        # 绑定a和node_ans
        for i in range(obs_switch_count):
            for j in range(phy_switch_count):
                model.addConstr((map_obsId_to_phyId[i,j] == 1) >> (phyId_for_obsId[i] == j),
                f"bind_obs_{i}_to_phy_{j}")
                
        # 计算延迟
        for k in range(obs_switch_count-1):
            for i in range(phy_switch_count):
                for j in range(phy_switch_count):
                    # 标记第k条链路的两端点
                    model.addConstr(obsEdge_to_phyEdge[k,i,j] == gp.and_(map_obsId_to_phyId[k,i], map_obsId_to_phyId[k+1,j]))
                    if i == j:
                        model.addConstr(obsEdge_to_phyEdge[k,i,j] == 0)
                        continue
                    # 获取延迟
                    path = self._network.get_shortest_path(i, j)
                    if path:
                        latency = self._network.get_path_latency(path)
                        model.addConstr((obsEdge_to_phyEdge[k,i,j] == 1) >> (latency_for_obsEdge[k] == latency))
                        
        # 设置总延迟目标
        latency_obj = gp.quicksum(latency_for_obsEdge[k] for k in range(obs_switch_count-1))
        
        # 目标：最小化总延迟
        model.setObjective(latency_obj, GRB.MINIMIZE)
        
        # 设置求解参数
        if self._config:
            model.setParam('TimeLimit', self._config.time_limit)
            model.setParam('MIPGap', self._config.gap_limit)
            model.setParam('OutputFlag', self._config.verbose)
            
        # 求解
        model.optimize()
        
        print("optimize for place_on_network complete")
        if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError("网络部署优化求解失败")
        obs_to_phy = {}
        # obs_to_phy记录从obs到phy的映射
        for obs_switch in range(obs_switch_count):
            for phy_switch in range(phy_switch_count):
                if map_obsId_to_phyId[obs_switch,phy_switch].x > 0.5:
                    obs_to_phy[obs_switch] = phy_switch
                    break
        self._best_objective = model.objVal
        self._status = OptimizationStatus.OPTIMAL
        return obs_to_phy
    
    def solve(self) -> Optional[Deployment]:
        """求解部署优化问题
        
        Returns:
            最优部署方案，如果未找到可行解则返回None
        """
        try:
            # 第一步：将任务部署到OBS上
            stage_to_mats = self._place_on_obs()
            # print("stage_to_mats:", stage_to_mats)
            # 第二步：将OBS部署到网络拓扑上
            obs_to_phy = self._place_on_network(stage_to_mats)
            # print("obs_to_phy:", obs_to_phy)

            # 创建部署方案
            deployment = Deployment(self._network, self._task)
            stage_per_switch = self._stage_count_per_switch_min()
            # 遍历obs_to_phy，将每个obs交换机映射到对应的物理交换机


            for obs_switch, phy_switch in obs_to_phy.items():
                # 计算该obs交换机的第一个stage编号
                first_stage = obs_switch * stage_per_switch
                # 遍历该obs交换机的所有stage
                for stage_id in range(first_stage, first_stage+stage_per_switch):
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
            return deployment
            
        except Exception as e:
            self._status = OptimizationStatus.ERROR
            raise RuntimeError(f"solve函数出错：{str(e)}")

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

    print(network.get_node_resource(0))
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
        optimizer = SpeedExact(network, task, config)
        deployment = optimizer.solve()
        if deployment:
            print("\n成功找到部署方案：")
            deployment.visualize()
            for vnf in task.get_vnfs():
                node = deployment.get_vnf_assignment(vnf).node_id
                stage = deployment.get_vnf_assignment(vnf).stage_id
                print(f"VNF {vnf} -> 节点 {node}  阶段 {stage}")
            print(f"总延迟：{deployment.calculate_total_latency()}")
        else:
            print("\n未找到可行的部署方案")
            
    except Exception as e:
        print(f"\n求解过程出错：{str(e)}")
