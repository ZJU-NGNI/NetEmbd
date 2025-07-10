"""MonPlan网络测量优化框架。

MonPlan是一个两阶段的网络测量优化框架：
1. 阶段一：测量点选择 (measure_point_selection)
2. 阶段二：测量数据收集 (measurement_data_collection)

主要组件：
- MonPlan: 主框架类，协调两阶段优化
- FlowGenerator: 流生成器，支持DCN和WAN网络
- Stage1算法: MonPlan, SpeedPlan, MtpPlan
- Stage2算法: Escala, Random, Optimal
- MonPlanEvaluator: 评估器，提供性能指标计算

Typical usage example:

    from netembd.optimize.MonPlan import MonPlan, FlowGenerator, MonPlanEvaluator
    from netembd.optimize.MonPlan.measurement_point_selection import MonPlanStage1
from netembd.optimize.MonPlan.data_collection_routing import EscalaStage2
    
    # 创建流生成器和算法
    flow_gen = FlowGenerator()
    stage1_alg = MonPlanStage1()
    stage2_alg = EscalaStage2()
    
    # 创建MonPlan实例
    monplan = MonPlan(flow_gen, stage1_alg, stage2_alg)
    
    # 执行优化
    result = monplan.optimize(network, num_flows=100)
    
    # 评估结果
    evaluator = MonPlanEvaluator()
    metrics = evaluator.evaluate_comprehensive(
        result.stage1_result, result.stage2_result, 
        result.flows, network
    )
"""

# 核心类
from .MonPlan import (
    MonPlan,
    MeasurementPoint,
    Stage1Result,
    Stage2Result,
    MonPlanResult,
    Stage1Algorithm,
    Stage2Algorithm
)

# 从network模块导入Flow和FlowGenerator
from netembd.network import Flow, FlowGenerator

# 测量点选择算法
from .measurement_point_selection import (
    BaseStage1Algorithm,
    MonPlanStage1,
    SpeedPlanStage1,
    MtpPlanStage1,
    MonPlan_LG_Stage1
)

# 数据收集路径算法
from .data_collection_routing import (
    BaseStage2Algorithm,
    EscalaStage2,
    RandomStage2,
    MonPlanStage2,
    IntPathStage2
)

# 评估模块
from .evaluation import (
    MonPlanEvaluator,
    Stage1Metrics,
    Stage2Metrics,
    ComprehensiveMetrics
)

__all__ = [
    # 核心类
    'MonPlan',
    'Flow',
    'MeasurementPoint',
    'Stage1Result',
    'Stage2Result',
    'MonPlanResult',
    'FlowGenerator',
    'Stage1Algorithm',
    'Stage2Algorithm',
    
    # 阶段一算法
    'BaseStage1Algorithm',
    'MonPlanStage1',
    'SpeedPlanStage1',
    'MtpPlanStage1',
    'MonPlan_LG_Stage1',
    
    # 阶段二算法
    'BaseStage2Algorithm',
    'EscalaStage2',
    'RandomStage2',
    'MonPlanStage2',
    'IntPathStage2',
    
    # 评估模块
    'MonPlanEvaluator',
    'Stage1Metrics',
    'Stage2Metrics',
    'ComprehensiveMetrics'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'MonPlan Team'
__description__ = 'Network Measurement Optimization Framework'

# 算法注册表，方便动态选择算法
STAGE1_ALGORITHMS = {
    'monplan': MonPlanStage1,
    'speedplan': SpeedPlanStage1,
    'mtpplan': MtpPlanStage1,
    'monplan_lg': MonPlan_LG_Stage1
}

STAGE2_ALGORITHMS = {
    'escala': EscalaStage2,
    'random': RandomStage2,
    'monplan': MonPlanStage2,
    'intpath': IntPathStage2
}

def create_stage1_algorithm(algorithm_name: str, **kwargs):
    """创建阶段一算法实例
    
    Args:
        algorithm_name: 算法名称 ('monplan', 'speedplan', 'mtpplan')
        **kwargs: 算法初始化参数
        
    Returns:
        算法实例
        
    Raises:
        ValueError: 不支持的算法名称
    """
    if algorithm_name.lower() not in STAGE1_ALGORITHMS:
        raise ValueError(f"不支持的阶段一算法: {algorithm_name}. "
                        f"支持的算法: {list(STAGE1_ALGORITHMS.keys())}")
    
    algorithm_class = STAGE1_ALGORITHMS[algorithm_name.lower()]
    return algorithm_class(**kwargs)

def create_stage2_algorithm(algorithm_name: str, **kwargs):
    """创建阶段二算法实例
    
    Args:
        algorithm_name: 算法名称 ('escala', 'random', 'optimal', 'monplan', 'intpath')
        **kwargs: 算法初始化参数
        
    Returns:
        算法实例
        
    Raises:
        ValueError: 不支持的算法名称
    """
    if algorithm_name.lower() not in STAGE2_ALGORITHMS:
        raise ValueError(f"不支持的阶段二算法: {algorithm_name}. "
                        f"支持的算法: {list(STAGE2_ALGORITHMS.keys())}")
    
    algorithm_class = STAGE2_ALGORITHMS[algorithm_name.lower()]
    return algorithm_class(**kwargs)

def create_monplan(network, stage1_algorithm: str, stage2_algorithm: str, 
                  num_flows: int = 100, stage1_kwargs=None, stage2_kwargs=None):
    """便捷函数：创建MonPlan实例
    
    Args:
        network: 网络拓扑实例
        stage1_algorithm: 阶段一算法名称
        stage2_algorithm: 阶段二算法名称
        num_flows: 流生成数量
        stage1_kwargs: 阶段一算法参数
        stage2_kwargs: 阶段二算法参数
        
    Returns:
        MonPlan实例
    """
    stage1_kwargs = stage1_kwargs or {}
    stage2_kwargs = stage2_kwargs or {}
    
    flow_generator = network.create_flow_generator(num_flows=num_flows)
    stage1_alg = create_stage1_algorithm(stage1_algorithm, **stage1_kwargs)
    stage2_alg = create_stage2_algorithm(stage2_algorithm, **stage2_kwargs)
    
    return MonPlan(flow_generator, stage1_alg, stage2_alg)