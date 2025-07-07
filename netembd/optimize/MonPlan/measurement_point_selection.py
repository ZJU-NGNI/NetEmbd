"""MonPlan阶段一：测量点选择算法实现。

此模块实现了测量点选择阶段的三种算法：
1. MonPlanStage1：最大化覆盖率 + 最小化控制跳数 + 最小化交换机数量
2. SpeedPlanStage1：最小化控制跳数 + 最小化交换机数量
3. MtpPlanStage1：最小化交换机数量

所有算法都使用Gurobi进行整数规划建模。
采用模板方法模式减少代码重复，提高可维护性。
"""

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from netembd.network import Network
from .MonPlan import Stage1Algorithm, Flow, MeasurementPoint, Stage1Result

@dataclass
class SolveContext:
    """求解上下文，封装算法求解所需的所有数据"""
    network: Network
    flows: List[Flow]
    programmable_nodes: List[int]
    control_nodes: List[int]
    node_flows: Dict[int, Set[int]]
    coverage_norm: float
    hop_norm: float
    switch_norm: float

@dataclass
class ModelVariables:
    """模型变量集合"""
    x_sketch: Dict[int, gp.Var]  # 节点是否部署sketch测量
    x_int: Dict[int, gp.Var]     # 节点是否部署INT测量
    y: Dict[int, gp.Var]         # 流是否被覆盖（仅MonPlan使用）

class BaseStage1Algorithm(Stage1Algorithm):
    """阶段一算法基类，使用模板方法模式提供通用求解流程"""
    
    def __init__(self, time_limit: int = 3600, gap_limit: float = 0.01, verbose: bool = False, 
                 normalize: bool = True, min_measurement_ratio: float = 1/3):
        """初始化基类
        
        Args:
            time_limit: 求解时间限制（秒）
            gap_limit: MIP Gap限制
            verbose: 是否输出详细日志
            normalize: 是否对目标函数进行归一化处理
            min_measurement_ratio: 最小测量点比例（相对于可编程交换机总数）
        """
        self.time_limit = time_limit
        self.gap_limit = gap_limit
        self.verbose = verbose
        self.normalize = normalize
        self.min_measurement_ratio = min_measurement_ratio
    
    def solve(self, network: Network, flows: List[Flow]) -> Stage1Result:
        """模板方法：定义统一的求解流程"""
        start_time = time.time()
        
        # 1. 准备求解上下文
        context = self._prepare_solve_context(network, flows)
        
        # 2. 创建Gurobi模型
        model = self._create_model()
        
        # 3. 创建决策变量
        variables = self._create_variables(model, context)
        
        # 4. 设置目标函数（子类实现）
        self._set_objective(model, variables, context)
        
        # 5. 添加约束条件
        self._add_constraints(model, variables, context)
        
        # 6. 求解模型并构建结果
        return self._solve_and_build_result(model, context, start_time)
    
    @abstractmethod
    def _set_objective(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """设置目标函数（子类必须实现）
        
        Args:
            model: Gurobi模型
            variables: 模型变量
            context: 求解上下文
        """
        pass
    
    def _prepare_solve_context(self, network: Network, flows: List[Flow]) -> SolveContext:
        """准备求解上下文"""
        programmable_nodes = self._get_programmable_nodes(network)
        control_nodes = self._get_control_nodes(network)
        node_flows = self._get_flow_node_coverage(flows)
        coverage_norm, hop_norm, switch_norm = self._normalize_objectives(
            network, flows, programmable_nodes, control_nodes)
        
        return SolveContext(
            network=network,
            flows=flows,
            programmable_nodes=programmable_nodes,
            control_nodes=control_nodes,
            node_flows=node_flows,
            coverage_norm=coverage_norm,
            hop_norm=hop_norm,
            switch_norm=switch_norm
        )
    
    def _create_model(self) -> gp.Model:
        """创建Gurobi模型"""
        model = gp.Model(f"{self.__class__.__name__}")
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.gap_limit)
        model.setParam('OutputFlag', 1 if self.verbose else 0)
        return model
    
    def _create_variables(self, model: gp.Model, context: SolveContext) -> ModelVariables:
        """创建决策变量"""
        x_sketch = {}
        x_int = {}
        y = {}
        
        # 为每个可编程节点创建测量变量
        for node in context.programmable_nodes:
            x_sketch[node] = model.addVar(vtype=GRB.BINARY, name=f'x_sketch_{node}')
            x_int[node] = model.addVar(vtype=GRB.BINARY, name=f'x_int_{node}')
        
        # 为每个流创建覆盖变量（MonPlan算法需要）
        for flow in context.flows:
            y[flow.flow_id] = model.addVar(vtype=GRB.BINARY, name=f'y_{flow.flow_id}')
        
        return ModelVariables(x_sketch=x_sketch, x_int=x_int, y=y)
    
    def _add_constraints(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """添加约束条件（子类可重写添加额外约束）"""
        self._add_min_measurement_constraint(model, variables, context)
    
    def _add_min_measurement_constraint(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """添加最小测量点约束"""
        min_measurement_points = max(1, int(len(context.programmable_nodes) * self.min_measurement_ratio))
        model.addConstr(
            gp.quicksum(variables.x_sketch[node] + variables.x_int[node] - 
                       variables.x_sketch[node] * variables.x_int[node] 
                       for node in context.programmable_nodes) >= min_measurement_points,
            name='min_measurement_points'
        )
    
    def _solve_and_build_result(self, model: gp.Model, context: SolveContext, start_time: float) -> Stage1Result:
        """求解模型并构建结果"""
        model.optimize()
        execution_time = time.time() - start_time
        
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return self._build_result(model, context.network, context.flows, 
                                    context.programmable_nodes, context.control_nodes,
                                    context.node_flows, execution_time)
        else:
            # 返回空结果
            return Stage1Result(
                measurement_points=[],
                flow_coverage=0.0,
                total_switches_used=0,
                total_control_hops=0,
                execution_time=execution_time,
                objective_value=float('inf')
            )
        
    def _normalize_objectives(self, network: Network, flows: List[Flow], 
                             programmable_nodes: List[int], control_nodes: List[int]):
        """计算归一化因子
        
        Args:
            network: 网络拓扑
            flows: 流列表
            programmable_nodes: 可编程节点列表
            control_nodes: 控制节点列表
            
        Returns:
            (coverage_norm, hop_norm, switch_norm): 三个目标的归一化因子
        """
        if not self.normalize:
            return 1.0, 1.0, 1.0
        
        # 覆盖率归一化：按流数量
        coverage_norm = len(flows)/2 if flows else 1.0
        
        # 交换机数量归一化：按可编程节点数
        switch_norm = len(programmable_nodes) if programmable_nodes else 1.0
        
        # 控制跳数归一化：按平均距离 × 节点数
        if programmable_nodes and control_nodes:
            total_distance = sum(self._calculate_control_distance(network, node, control_nodes) 
                              for node in programmable_nodes)
            avg_distance = total_distance / len(programmable_nodes) if total_distance > 0 else 1.0
            hop_norm = avg_distance * len(programmable_nodes)
        else:
            hop_norm = 1.0
        
        return coverage_norm, hop_norm, switch_norm
    
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
    
    def _get_programmable_nodes(self, network: Network) -> List[int]:
        """获取可编程节点列表
        
        Args:
            network: 网络拓扑
            
        Returns:
            可编程节点ID列表
        """
        programmable_nodes = []
        for node in network.get_nodes():
            resource = network.get_node_resource(node)
            if resource.programmable and not resource.control_node:
                programmable_nodes.append(node)
        return programmable_nodes
    
    def _calculate_control_distance(self, network: Network, node: int, control_nodes: List[int]) -> int:
        """计算节点到最近控制节点的跳数
        
        Args:
            network: 网络拓扑
            node: 节点ID
            control_nodes: 控制节点列表
            
        Returns:
            到最近控制节点的跳数
        """
        min_distance = float('inf')
        for control_node in control_nodes:
            path = network.get_shortest_path(node, control_node)
            if path is not None:
                distance = len(path) - 1  # 跳数 = 路径长度 - 1
                min_distance = min(min_distance, distance)
        return int(min_distance) if min_distance != float('inf') else 0
    
    def _get_flow_node_coverage(self, flows: List[Flow]) -> Dict[int, Set[int]]:
        """获取每个节点覆盖的流集合
        
        Args:
            flows: 流列表
            
        Returns:
            节点ID到流ID集合的映射
        """
        node_flows = {}
        for flow in flows:
            for node in flow.path:
                if node not in node_flows:
                    node_flows[node] = set()
                node_flows[node].add(flow.flow_id)
        return node_flows
    
    def _build_result(self, model: gp.Model, network: Network, flows: List[Flow], 
                     programmable_nodes: List[int], control_nodes: List[int],
                     node_flows: Dict[int, Set[int]], execution_time: float) -> Stage1Result:
        """构建求解结果
        
        Args:
            model: Gurobi模型
            network: 网络拓扑
            flows: 流列表
            programmable_nodes: 可编程节点列表
            control_nodes: 控制节点列表
            node_flows: 节点到流的映射
            execution_time: 执行时间
            
        Returns:
            阶段一结果
        """
        measurement_points = []
        total_switches_used = 0
        total_control_hops = 0
        covered_flows = set()
        
        # 提取选中的测量点
        for node in programmable_nodes:
            x_sketch = model.getVarByName(f'x_sketch_{node}')
            x_int = model.getVarByName(f'x_int_{node}')
            
            measurement_types = []
            if x_sketch and x_sketch.X > 0.5:
                measurement_types.append('sketch')
            if x_int and x_int.X > 0.5:
                measurement_types.append('INT')
            
            if measurement_types:
                # 计算控制距离
                control_distance = self._calculate_control_distance(network, node, control_nodes)
                
                # 获取覆盖的流（根据算法类型决定）
                if isinstance(self, MonPlanStage1):
                    # MonPlan算法：使用y变量确定实际覆盖的流
                    node_covered_flows = []
                    for flow in flows:
                        if node in flow.path:
                            y_var = model.getVarByName(f'y_{flow.flow_id}')
                            if y_var and y_var.X > 0.5:
                                node_covered_flows.append(flow.flow_id)
                                covered_flows.add(flow.flow_id)
                else:
                    # 其他算法：基于节点路径覆盖计算
                    node_covered_flows = list(node_flows.get(node, set()))
                    covered_flows.update(node_covered_flows)
                
                measurement_point = MeasurementPoint(
                    node_id=node,
                    measurement_types=measurement_types,
                    covered_flows=node_covered_flows,
                    control_node_distance=control_distance
                )
                measurement_points.append(measurement_point)
                
                total_switches_used += 1
                total_control_hops += control_distance
        
        # 计算流覆盖率
        flow_coverage = len(covered_flows) / len(flows) if flows else 0.0
        
        return Stage1Result(
            measurement_points=measurement_points,
            flow_coverage=flow_coverage,
            total_switches_used=total_switches_used,
            total_control_hops=total_control_hops,
            execution_time=execution_time,
            objective_value=model.ObjVal if model.Status == GRB.OPTIMAL else float('inf')
        )

class MonPlanStage1(BaseStage1Algorithm):
    """MonPlan阶段一算法：最大化覆盖率 + 最小化控制跳数 + 最小化交换机数量"""
    
    def __init__(self, coverage_weight: float = 0.5, hop_weight: float = 0.25, 
                 switch_weight: float = 0.25, **kwargs):
        """初始化MonPlan算法
        
        Args:
            coverage_weight: 覆盖率权重（归一化后，建议使用1.0作为基准）
            hop_weight: 控制跳数权重（归一化后，建议使用1.0作为基准）
            switch_weight: 交换机数量权重（归一化后，建议使用1.0作为基准）
        """
        super().__init__(**kwargs)
        self.coverage_weight = coverage_weight
        self.hop_weight = hop_weight
        self.switch_weight = switch_weight
    
    def _set_objective(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """设置MonPlan目标函数：最大化覆盖率 - 最小化控制跳数 - 最小化交换机数量"""
        # 目标函数：最大化覆盖率 - 最小化控制跳数 - 最小化交换机数量
        coverage_term = self.coverage_weight * gp.quicksum(variables.y[flow.flow_id] for flow in context.flows) / context.coverage_norm
        
        hop_term = self.hop_weight * gp.quicksum(
            self._calculate_control_distance(context.network, node, context.control_nodes) * 
            (variables.x_sketch[node] + variables.x_int[node])
            for node in context.programmable_nodes
        ) / context.hop_norm
        
        switch_term = self.switch_weight * gp.quicksum(
            variables.x_sketch[node] + variables.x_int[node] - variables.x_sketch[node] * variables.x_int[node]
            for node in context.programmable_nodes
        ) / context.switch_norm
        
        model.setObjective(coverage_term - hop_term - switch_term, GRB.MAXIMIZE)
    
    def _add_constraints(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """添加MonPlan特有的约束条件"""
        # 调用基类的约束
        super()._add_constraints(model, variables, context)
        
        # 流覆盖约束：如果流被覆盖，则其路径上至少有一个测量点
        for flow in context.flows:
            path_nodes = [node for node in flow.path if node in context.programmable_nodes]
            if path_nodes:
                # 根据流大小选择测量类型
                if flow.measurement_type == 'sketch':
                    model.addConstr(
                        variables.y[flow.flow_id] <= gp.quicksum(variables.x_sketch[node] for node in path_nodes),
                        name=f'coverage_sketch_{flow.flow_id}'
                    )
                else:  # INT
                    model.addConstr(
                        variables.y[flow.flow_id] <= gp.quicksum(variables.x_int[node] for node in path_nodes),
                        name=f'coverage_int_{flow.flow_id}'
                    )

class SpeedPlanStage1(BaseStage1Algorithm):
    """SpeedPlan阶段一算法：最小化控制跳数 + 最小化交换机数量"""
    
    def __init__(self, hop_weight: float = 1/2, switch_weight: float = 1/2, **kwargs):
        """初始化SpeedPlan算法
        
        Args:
            hop_weight: 控制跳数权重（归一化后，建议使用1.0作为基准）
            switch_weight: 交换机数量权重（归一化后，建议使用1.0作为基准）
        """
        super().__init__(**kwargs)
        self.hop_weight = hop_weight
        self.switch_weight = switch_weight
    
    def _set_objective(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """设置SpeedPlan目标函数：最小化控制跳数 + 最小化交换机数量"""
        # 目标函数：最小化控制跳数 + 最小化交换机数量
        hop_term = self.hop_weight * gp.quicksum(
            self._calculate_control_distance(context.network, node, context.control_nodes) * 
            (variables.x_sketch[node] + variables.x_int[node])
            for node in context.programmable_nodes
        ) / context.hop_norm
        
        switch_term = self.switch_weight * gp.quicksum(
            variables.x_sketch[node] + variables.x_int[node] - variables.x_sketch[node] * variables.x_int[node]
            for node in context.programmable_nodes
        ) / context.switch_norm
        
        model.setObjective(hop_term + switch_term, GRB.MINIMIZE)

class MtpPlanStage1(BaseStage1Algorithm):
    """MtpPlan阶段一算法：最小化交换机数量"""
    
    def __init__(self, **kwargs):
        """初始化MtpPlan算法
        
        Args:
            **kwargs: 基类参数，包括min_measurement_ratio等
        """
        super().__init__(**kwargs)
    
    def _set_objective(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """设置MtpPlan目标函数：最小化交换机数量"""
        # 目标函数：最小化交换机数量
        model.setObjective(
            gp.quicksum(variables.x_sketch[node] + variables.x_int[node] - variables.x_sketch[node] * variables.x_int[node]
                       for node in context.programmable_nodes) / context.switch_norm,
            GRB.MINIMIZE
        )

