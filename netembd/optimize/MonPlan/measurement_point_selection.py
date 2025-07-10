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
    
    def __init__(self, coverage_weight: float = 0.8, hop_weight: float = 0.2, 
                 switch_weight: float = 0, **kwargs):
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

class MonPlan_LG_Stage1(BaseStage1Algorithm):
    """MonPlan拉格朗日对偶阶段一算法：使用子梯度优化求解拉格朗日对偶问题"""
    
    def __init__(self, alpha: float = 0.6, max_iterations: int = 100, 
                 lambda_init: float = 0.5, epsilon: float = 1e-6, **kwargs):
        """初始化MonPlan拉格朗日对偶算法
        
        Args:
            alpha: 覆盖奖励参数
            max_iterations: 最大迭代次数
            lambda_init: 拉格朗日乘数初始值
            epsilon: 数值稳定性参数
            **kwargs: 基类参数
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
        self.epsilon = epsilon
    
    def solve(self, network: Network, flows: List[Flow]) -> Stage1Result:
        """使用拉格朗日对偶方法求解测量点选择问题"""
        start_time = time.time()
        
        # 准备求解上下文
        context = self._prepare_solve_context(network, flows)
        
        # 执行子梯度优化
        best_solution = self._subgradient_optimization(context)
        
        # 构建结果
        execution_time = time.time() - start_time
        return self._build_lg_result(best_solution, context, execution_time)
    
    def _set_objective(self, model: gp.Model, variables: ModelVariables, context: SolveContext):
        """拉格朗日对偶算法不使用Gurobi模型，此方法为空实现"""
        pass
    
    def _subgradient_optimization(self, context: SolveContext) -> Dict:
        """执行子梯度优化算法（算法2）
        
        Args:
            context: 求解上下文
            
        Returns:
            最优解字典，包含up, wf, zpc等变量
        """
        # 初始化拉格朗日乘数：λ_f^(0) = 0.5（算法2第1行）
        lambda_f = {flow.flow_id: self.lambda_init for flow in context.flows}
        best_solution = None
        L_best = -float('inf')
        lambda_best = lambda_f.copy()
        
        # 构建流路径映射（G[f]）
        flow_paths = self._build_flow_paths(context)
        
        # 迭代优化（算法2第3-13行）
        for t in range(self.max_iterations):
            # 求解L(λ^(t))（算法2第4行）
            L_lambda_t, up, wf, zpc, xp, yp = self._solve_L(lambda_f, flow_paths, context)
            
            # 计算子梯度（算法2第5行）
            g_f_t = self._compute_subgradient(up, wf, flow_paths, context)
            
            # 计算梯度范数（算法2第6行）
            g_f_t_list = list(g_f_t.values())
            g_f_t_norm = sum(g**2 for g in g_f_t_list) ** 0.5
            
            # 计算步长（算法2第7行）
            theta_t = (1 / (t + 1)**0.5) * abs(L_lambda_t) / max(g_f_t_norm**2, self.epsilon)
            
            # 更新拉格朗日乘数（算法2第8-9行）
            for flow in context.flows:
                lambda_f[flow.flow_id] = max(0, min(2, lambda_f[flow.flow_id] + theta_t * g_f_t[flow.flow_id]))
            
            # 更新最优解（算法2第10-12行）
            if L_lambda_t > L_best:
                L_best = L_lambda_t
                lambda_best = lambda_f.copy()
                best_solution = {
                    'up': up.copy(),
                    'wf': wf.copy(),
                    'zpc': zpc.copy(),
                    'xp': xp.copy(),
                    'yp': yp.copy(),
                    'objective': L_lambda_t
                }
        
        return best_solution if best_solution else {
            'up': {}, 'wf': {}, 'zpc': {}, 'xp': {}, 'yp': {}, 'objective': float('inf')
        }
    
    def _solve_L(self, lambda_f: Dict[int, float], flow_paths: Dict[int, List[int]], 
                 context: SolveContext) -> Tuple[float, Dict, Dict, Dict, Dict, Dict]:
        """求解L(λ)子问题，严格实现约束条件(2)-(7)
        
        Args:
            lambda_f: 拉格朗日乘数字典
            flow_paths: 流路径映射
            context: 求解上下文
            
        Returns:
            (L值, up变量, wf变量, zpc变量, xp变量, yp变量)
        """
        up = {p: 0 for p in context.programmable_nodes}
        wf = {flow.flow_id: 0 for flow in context.flows}
        zpc = {}
        # 新增：测量类型变量
        xp = {p: 0 for p in context.programmable_nodes}  # sketch测量
        yp = {p: 0 for p in context.programmable_nodes}  # INT测量
        
        # 对每个可编程节点决策是否激活及测量类型
        for p in context.programmable_nodes:
            # 计算β_p：通过节点p的所有流的λ值之和
            beta_p = sum(lambda_f[flow.flow_id] for flow in context.flows 
                        if p in flow_paths.get(flow.flow_id, []))
            
            # 计算γ_p：(1-α) * 到最近控制节点的距离
            gamma_p = (1 - self.alpha) * self._calculate_control_distance(
                context.network, p, context.control_nodes)
            
            # 如果β_p > γ_p，激活节点p
            if beta_p > gamma_p:
                up[p] = 1
                
                # 决策测量类型：优先选择sketch（成本更低）
                # 约束(2): up >= xp, 约束(3): up >= yp, 约束(4): up <= xp + yp
                # 为简化，选择sketch测量
                xp[p] = 1
                yp[p] = 0
                
                # 分配到最近的控制节点
                c_star = min(context.control_nodes, 
                           key=lambda c: self._calculate_control_distance(
                               context.network, p, [c]))
                zpc[(p, c_star)] = 1
        
        # 验证约束条件(2)-(4)
        for p in context.programmable_nodes:
            # 约束(2): up >= xp
            assert up[p] >= xp[p], f"约束(2)违反：节点{p}, up={up[p]}, xp={xp[p]}"
            # 约束(3): up >= yp  
            assert up[p] >= yp[p], f"约束(3)违反：节点{p}, up={up[p]}, yp={yp[p]}"
            # 约束(4): up <= xp + yp
            assert up[p] <= xp[p] + yp[p], f"约束(4)违反：节点{p}, up={up[p]}, xp+yp={xp[p]+yp[p]}"
        
        # 对每个流决策是否覆盖
        for flow in context.flows:
            # 约束(5): wf <= Σ_{p∈Pf} up
            path_nodes = flow_paths.get(flow.flow_id, [])
            sum_up_on_path = sum(up.get(p, 0) for p in path_nodes)
            
            # 如果流路径上存在激活节点，则可以覆盖流
            if sum_up_on_path >= 1:
                wf[flow.flow_id] = 1
            else:
                wf[flow.flow_id] = 0
                
            # 验证约束(5): wf <= Σ_{p∈Pf} up
            assert wf[flow.flow_id] <= sum_up_on_path, f"约束(5)违反：流{flow.flow_id}"
        
        # 验证约束(6): Σ_c zpc = up
        for p in context.programmable_nodes:
            sum_zpc = sum(zpc.get((p, c), 0) for c in context.control_nodes)
            assert sum_zpc == up[p], f"约束(6)违反：节点{p}, sum_zpc={sum_zpc}, up={up[p]}"
        
        # 计算L(λ)值：L(λ) = Σ_p u_p(β_p + γ_p) + Σ_f w_f(-α - λ_f)
        L = 0
        for p in context.programmable_nodes:
            if up[p] == 1:
                beta_p = sum(lambda_f[flow.flow_id] for flow in context.flows 
                           if p in flow_paths.get(flow.flow_id, []))
                gamma_p = (1 - self.alpha) * self._calculate_control_distance(
                    context.network, p, context.control_nodes)
                L += up[p] * (beta_p + gamma_p)
        
        for flow in context.flows:
            L += wf[flow.flow_id] * (-self.alpha - lambda_f[flow.flow_id])
        
        return L, up, wf, zpc, xp, yp
    
    def _compute_subgradient(self, up: Dict, wf: Dict, flow_paths: Dict[int, List[int]], 
                           context: SolveContext) -> Dict[int, float]:
        """计算子梯度：g_f = Σ_{p∈P_f} u_p* - w_f*（算法2第5行）
        
        Args:
            up: 节点激活变量
            wf: 流覆盖变量
            flow_paths: 流路径映射
            context: 求解上下文
            
        Returns:
            子梯度字典
        """
        g_f_t = {}
        for flow in context.flows:
            # g_f = Σ_{p∈P_f} u_p* - w_f*
            path_nodes = flow_paths.get(flow.flow_id, [])
            g_f_t[flow.flow_id] = sum(up.get(p, 0) for p in path_nodes) - wf.get(flow.flow_id, 0)
        return g_f_t
    
    def _build_flow_paths(self, context: SolveContext) -> Dict[int, List[int]]:
        """构建流路径映射
        
        Args:
            context: 求解上下文
            
        Returns:
            流ID到路径节点列表的映射
        """
        flow_paths = {}
        for flow in context.flows:
            # 只保留可编程节点
            path_nodes = [node for node in flow.path if node in context.programmable_nodes]
            flow_paths[flow.flow_id] = path_nodes
        return flow_paths
    
    def _build_lg_result(self, solution: Dict, context: SolveContext, 
                        execution_time: float) -> Stage1Result:
        """构建拉格朗日对偶算法的求解结果，严格基于约束条件(2)-(7)
        
        Args:
            solution: 最优解字典
            context: 求解上下文
            execution_time: 执行时间
            
        Returns:
            阶段一结果
        """
        measurement_points = []
        total_switches_used = 0
        total_control_hops = 0
        covered_flows = set()
        
        up = solution.get('up', {})
        wf = solution.get('wf', {})
        zpc = solution.get('zpc', {})
        xp = solution.get('xp', {})  # sketch测量变量
        yp = solution.get('yp', {})  # INT测量变量
        
        # 构建测量点
        for node in context.programmable_nodes:
            if up.get(node, 0) > 0.5:
                # 根据xp和yp变量确定测量类型
                measurement_types = []
                if xp.get(node, 0) > 0.5:
                    measurement_types.append('sketch')
                if yp.get(node, 0) > 0.5:
                    measurement_types.append('INT')
                
                # 验证约束条件
                assert up[node] >= xp.get(node, 0), f"约束(2)违反：节点{node}"
                assert up[node] >= yp.get(node, 0), f"约束(3)违反：节点{node}"
                assert up[node] <= xp.get(node, 0) + yp.get(node, 0), f"约束(4)违反：节点{node}"
                
                # 计算控制距离
                control_distance = self._calculate_control_distance(
                    context.network, node, context.control_nodes)
                
                # 获取覆盖的流
                node_covered_flows = []
                for flow in context.flows:
                    if node in flow.path and wf.get(flow.flow_id, 0) > 0.5:
                        node_covered_flows.append(flow.flow_id)
                        covered_flows.add(flow.flow_id)
                
                measurement_point = MeasurementPoint(
                    node_id=node,
                    measurement_types=measurement_types,
                    covered_flows=node_covered_flows,
                    control_node_distance=control_distance
                )
                measurement_points.append(measurement_point)
                
                total_switches_used += 1
                total_control_hops += control_distance
        
        # 验证流覆盖约束(5)
        for flow in context.flows:
            if wf.get(flow.flow_id, 0) > 0.5:
                # 检查流路径上是否有激活的测量点
                path_has_measurement = any(up.get(node, 0) > 0.5 for node in flow.path 
                                         if node in context.programmable_nodes)
                assert path_has_measurement, f"约束(5)违反：流{flow.flow_id}被覆盖但路径上无激活节点"
        
        # 计算流覆盖率
        flow_coverage = len(covered_flows) / len(context.flows) if context.flows else 0.0
        
        return Stage1Result(
            measurement_points=measurement_points,
            flow_coverage=flow_coverage,
            total_switches_used=total_switches_used,
            total_control_hops=total_control_hops,
            execution_time=execution_time,
            objective_value=solution.get('objective', float('inf'))
        )

