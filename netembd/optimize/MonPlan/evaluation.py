"""MonPlan评估模块

此模块提供了MonPlan框架的评估指标计算功能，包括：
1. 阶段一评估：流覆盖率、执行时间、交换机使用数量
2. 阶段二评估：跳数、拥塞率、执行时间
3. 综合评估：端到端性能指标
4. 对比分析：多算法性能对比

Typical usage example:

    from netembd.optimize.MonPlan.evaluation import (
        Stage1Evaluator, Stage2Evaluator, ComprehensiveEvaluator
    )
    
    stage1_evaluator = Stage1Evaluator()
    stage2_evaluator = Stage2Evaluator()
    comprehensive_evaluator = ComprehensiveEvaluator(stage1_evaluator, stage2_evaluator)
    
    stage1_metrics = stage1_evaluator.evaluate(stage1_result, flows, network)
    comparison = comprehensive_evaluator.compare_algorithms(results_dict)
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import json
import csv
from collections import defaultdict
from abc import ABC, abstractmethod

from netembd.network import Network
from .MonPlan import Flow, Stage1Result, Stage2Result, MeasurementPoint

@dataclass
class Stage1Metrics:
    """阶段一评估指标"""
    flow_coverage_rate: float  # 流覆盖率 (0-1)
    total_flows: int  # 总流数
    covered_flows: int  # 覆盖的流数
    total_switches_used: int  # 使用的交换机数量
    total_programmable_switches: int  # 总可编程交换机数量
    switch_utilization_rate: float  # 交换机利用率
    total_control_hops: int  # 总控制跳数
    average_control_hops: float  # 平均控制跳数
    execution_time: float  # 执行时间（秒）
    objective_value: float  # 目标函数值
    algorithm_name: str  # 算法名称

@dataclass
class Stage2Metrics:
    """阶段二评估指标"""
    total_hop_count: int  # 总跳数
    average_hop_count: float  # 平均跳数
    congestion_rate: float  # 拥塞率 (0-1)
    total_data_paths: int  # 总数据路径数
    execution_time: float  # 执行时间（秒）
    algorithm_name: str  # 算法名称
    max_path_length: int  # 最长路径长度
    min_path_length: int  # 最短路径长度

@dataclass
class ComprehensiveMetrics:
    """综合评估指标"""
    stage1_metrics: Stage1Metrics
    stage2_metrics: Stage2Metrics
    total_execution_time: float  # 总执行时间
    end_to_end_latency: float  # 端到端延迟
    system_efficiency: float  # 系统效率指标
    algorithm_combination: str  # 算法组合名称

class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(self):
        """初始化评估器"""
        self.evaluation_history = []
    
    def clear_history(self) -> None:
        """清空评估历史"""
        self.evaluation_history.clear()
    
    def export_results(self, filename: str, format: str = "json") -> None:
        """导出评估结果
        
        Args:
            filename: 输出文件名
            format: 输出格式 ('json' 或 'csv')
        """
        if not self.evaluation_history:
            print("没有评估结果可导出")
            return
        
        if format.lower() == "json":
            self._export_json(filename)
        elif format.lower() == "csv":
            self._export_csv(filename)
        else:
            raise ValueError("不支持的格式，请使用 'json' 或 'csv'")
    
    def _export_json(self, filename: str) -> None:
        """导出JSON格式结果"""
        data = [asdict(metrics) for metrics in self.evaluation_history]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"结果已导出到 {filename}")
    
    @abstractmethod
    def _export_csv(self, filename: str) -> None:
        """导出CSV格式结果"""
        pass

class Stage1Evaluator(BaseEvaluator):
    """阶段一评估器"""
    
    def __init__(self):
        """初始化阶段一评估器"""
        super().__init__()
    
    def evaluate(self, stage1_result: Stage1Result, flows: List[Flow], 
                network: Network, algorithm_name: str = "Unknown") -> Stage1Metrics:
        """评估阶段一结果
        
        Args:
            stage1_result: 阶段一结果
            flows: 流列表
            network: 网络拓扑
            algorithm_name: 算法名称
            
        Returns:
            阶段一评估指标
        """
        # 计算可编程交换机总数
        total_programmable = 0
        for node in network.get_nodes():
            resource = network.get_node_resource(node)
            if resource.programmable and not resource.control_node:
                total_programmable += 1
        
        # 计算覆盖的流数
        covered_flows = set()
        for mp in stage1_result.measurement_points:
            covered_flows.update(mp.covered_flows)
        
        # 计算交换机利用率
        switch_utilization = (stage1_result.total_switches_used / total_programmable 
                            if total_programmable > 0 else 0.0)
        
        # 计算平均控制跳数
        avg_control_hops = (stage1_result.total_control_hops / stage1_result.total_switches_used 
                          if stage1_result.total_switches_used > 0 else 0.0)
        
        metrics = Stage1Metrics(
            flow_coverage_rate=stage1_result.flow_coverage,
            total_flows=len(flows),
            covered_flows=len(covered_flows),
            total_switches_used=stage1_result.total_switches_used,
            total_programmable_switches=total_programmable,
            switch_utilization_rate=switch_utilization,
            total_control_hops=stage1_result.total_control_hops,
            average_control_hops=avg_control_hops,
            execution_time=stage1_result.execution_time,
            objective_value=stage1_result.objective_value,
            algorithm_name=algorithm_name
        )
        
        # 记录评估历史
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def _export_csv(self, filename: str) -> None:
        """导出CSV格式结果"""
        if not self.evaluation_history:
            return
        
        # 展平数据结构
        flattened_data = []
        for metrics in self.evaluation_history:
            row = {
                "algorithm_name": metrics.algorithm_name,
                "flow_coverage_rate": metrics.flow_coverage_rate,
                "total_flows": metrics.total_flows,
                "covered_flows": metrics.covered_flows,
                "total_switches_used": metrics.total_switches_used,
                "total_programmable_switches": metrics.total_programmable_switches,
                "switch_utilization_rate": metrics.switch_utilization_rate,
                "total_control_hops": metrics.total_control_hops,
                "average_control_hops": metrics.average_control_hops,
                "execution_time": metrics.execution_time,
                "objective_value": metrics.objective_value,
            }
            flattened_data.append(row)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
        
        print(f"结果已导出到 {filename}")
    
    def compare_algorithms(self, results: Dict[str, Stage1Metrics]) -> Dict[str, Any]:
        """对比多个阶段一算法的性能
        
        Args:
            results: 算法名称到阶段一指标的映射
            
        Returns:
            对比分析结果
        """
        if not results:
            return {}
        
        comparison = {
            "algorithms": list(results.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "best_algorithm": {},
            "summary": {}
        }
        
        # 提取各项指标
        metrics_data = {
            "flow_coverage_rate": {},
            "switch_utilization_rate": {},
            "execution_time": {},
            "average_control_hops": {},
            "objective_value": {}
        }
        
        for alg_name, metrics in results.items():
            metrics_data["flow_coverage_rate"][alg_name] = metrics.flow_coverage_rate
            metrics_data["switch_utilization_rate"][alg_name] = metrics.switch_utilization_rate
            metrics_data["execution_time"][alg_name] = metrics.execution_time
            metrics_data["average_control_hops"][alg_name] = metrics.average_control_hops
            metrics_data["objective_value"][alg_name] = metrics.objective_value
        
        comparison["metrics_comparison"] = metrics_data
        
        # 计算排名（越大越好的指标）
        for metric in ["flow_coverage_rate"]:
            sorted_algs = sorted(metrics_data[metric].items(), key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric] = [alg for alg, _ in sorted_algs]
            comparison["best_algorithm"][metric] = sorted_algs[0][0]
        
        # 计算排名（越小越好的指标）
        for metric in ["switch_utilization_rate", "execution_time", "average_control_hops"]:
            sorted_algs = sorted(metrics_data[metric].items(), key=lambda x: x[1])
            comparison["rankings"][metric] = [alg for alg, _ in sorted_algs]
            comparison["best_algorithm"][metric] = sorted_algs[0][0]
        
        return comparison
    
class Stage2Evaluator(BaseEvaluator):
    """阶段二评估器"""
    
    def __init__(self):
        """初始化阶段二评估器"""
        super().__init__()
    
    def evaluate(self, stage2_result: Stage2Result, 
                algorithm_name: str = "Unknown") -> Stage2Metrics:
        """评估阶段二结果
        
        Args:
            stage2_result: 阶段二结果
            algorithm_name: 算法名称
            
        Returns:
            阶段二评估指标
        """
        # 计算路径统计信息
        path_lengths = []
        total_paths = 0
        
        for measurement_node, paths in stage2_result.data_paths.items():
            total_paths += len(paths)
            for path in paths:
                path_lengths.append(len(path) - 1)  # 跳数 = 路径长度 - 1
        
        avg_hop_count = (stage2_result.total_hop_count / total_paths 
                        if total_paths > 0 else 0.0)
        
        max_path_length = max(path_lengths) if path_lengths else 0
        min_path_length = min(path_lengths) if path_lengths else 0
        
        metrics = Stage2Metrics(
            total_hop_count=stage2_result.total_hop_count,
            average_hop_count=avg_hop_count,
            congestion_rate=stage2_result.congestion_rate,
            total_data_paths=total_paths,
            execution_time=stage2_result.execution_time,
            algorithm_name=algorithm_name,
            max_path_length=max_path_length,
            min_path_length=min_path_length
        )
        
        # 记录评估历史
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def _export_csv(self, filename: str) -> None:
        """导出CSV格式结果"""
        if not self.evaluation_history:
            return
        
        # 展平数据结构
        flattened_data = []
        for metrics in self.evaluation_history:
            row = {
                "algorithm_name": metrics.algorithm_name,
                "total_hop_count": metrics.total_hop_count,
                "average_hop_count": metrics.average_hop_count,
                "congestion_rate": metrics.congestion_rate,
                "total_data_paths": metrics.total_data_paths,
                "execution_time": metrics.execution_time,
                "max_path_length": metrics.max_path_length,
                "min_path_length": metrics.min_path_length,
            }
            flattened_data.append(row)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
        
        print(f"结果已导出到 {filename}")
    
    def compare_algorithms(self, results: Dict[str, Stage2Metrics]) -> Dict[str, Any]:
        """对比多个阶段二算法的性能
        
        Args:
            results: 算法名称到阶段二指标的映射
            
        Returns:
            对比分析结果
        """
        if not results:
            return {}
        
        comparison = {
            "algorithms": list(results.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "best_algorithm": {},
            "summary": {}
        }
        
        # 提取各项指标
        metrics_data = {
            "congestion_rate": {},
            "average_hop_count": {},
            "execution_time": {},
            "max_path_length": {}
        }
        
        for alg_name, metrics in results.items():
            metrics_data["congestion_rate"][alg_name] = metrics.congestion_rate
            metrics_data["average_hop_count"][alg_name] = metrics.average_hop_count
            metrics_data["execution_time"][alg_name] = metrics.execution_time
            metrics_data["max_path_length"][alg_name] = metrics.max_path_length
        
        comparison["metrics_comparison"] = metrics_data
        
        # 计算排名（越小越好的指标）
        for metric in ["congestion_rate", "average_hop_count", "execution_time", "max_path_length"]:
            sorted_algs = sorted(metrics_data[metric].items(), key=lambda x: x[1])
            comparison["rankings"][metric] = [alg for alg, _ in sorted_algs]
            comparison["best_algorithm"][metric] = sorted_algs[0][0]
        
        return comparison
    
class ComprehensiveEvaluator(BaseEvaluator):
    """综合评估器"""
    
    def __init__(self, stage1_evaluator: Stage1Evaluator, stage2_evaluator: Stage2Evaluator):
        """初始化综合评估器
        
        Args:
            stage1_evaluator: 阶段一评估器
            stage2_evaluator: 阶段二评估器
        """
        super().__init__()
        self.stage1_evaluator = stage1_evaluator
        self.stage2_evaluator = stage2_evaluator
    
    def evaluate_comprehensive(self, stage1_result: Stage1Result, stage2_result: Stage2Result,
                             flows: List[Flow], network: Network,
                             stage1_algorithm: str = "Unknown", 
                             stage2_algorithm: str = "Unknown") -> ComprehensiveMetrics:
        """综合评估两阶段结果
        
        Args:
            stage1_result: 阶段一结果
            stage2_result: 阶段二结果
            flows: 流列表
            network: 网络拓扑
            stage1_algorithm: 阶段一算法名称
            stage2_algorithm: 阶段二算法名称
            
        Returns:
            综合评估指标
        """
        stage1_metrics = self.stage1_evaluator.evaluate(stage1_result, flows, network, stage1_algorithm)
        stage2_metrics = self.stage2_evaluator.evaluate(stage2_result, stage2_algorithm)
        
        # 计算端到端延迟（简化计算）
        end_to_end_latency = self._calculate_end_to_end_latency(
            stage1_result, stage2_result, network
        )
        
        # 计算系统效率（综合指标）
        system_efficiency = self._calculate_system_efficiency(
            stage1_metrics, stage2_metrics
        )
        
        total_execution_time = stage1_result.execution_time + stage2_result.execution_time
        algorithm_combination = f"{stage1_algorithm}+{stage2_algorithm}"
        
        comprehensive_metrics = ComprehensiveMetrics(
            stage1_metrics=stage1_metrics,
            stage2_metrics=stage2_metrics,
            total_execution_time=total_execution_time,
            end_to_end_latency=end_to_end_latency,
            system_efficiency=system_efficiency,
            algorithm_combination=algorithm_combination
        )
        
        # 记录评估历史
        self.evaluation_history.append(comprehensive_metrics)
        
        return comprehensive_metrics
    
    def _export_csv(self, filename: str) -> None:
        """导出CSV格式结果"""
        if not self.evaluation_history:
            return
        
        # 展平数据结构
        flattened_data = []
        for metrics in self.evaluation_history:
            row = {
                "algorithm_combination": metrics.algorithm_combination,
                "total_execution_time": metrics.total_execution_time,
                "end_to_end_latency": metrics.end_to_end_latency,
                "system_efficiency": metrics.system_efficiency,
                # 阶段一指标
                "stage1_flow_coverage_rate": metrics.stage1_metrics.flow_coverage_rate,
                "stage1_switch_utilization_rate": metrics.stage1_metrics.switch_utilization_rate,
                "stage1_execution_time": metrics.stage1_metrics.execution_time,
                # 阶段二指标
                "stage2_congestion_rate": metrics.stage2_metrics.congestion_rate,
                "stage2_total_hop_count": metrics.stage2_metrics.total_hop_count,
                "stage2_execution_time": metrics.stage2_metrics.execution_time,
            }
            flattened_data.append(row)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
        
        print(f"结果已导出到 {filename}")
    
    def compare_algorithms(self, results: Dict[str, ComprehensiveMetrics]) -> Dict[str, Any]:
        """对比多个算法的性能
        
        Args:
            results: 算法名称到综合指标的映射
            
        Returns:
            对比分析结果
        """
        if not results:
            return {}
        
        comparison = {
            "algorithms": list(results.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "best_algorithm": {},
            "summary": {}
        }
        
        # 提取各项指标
        metrics_data = {
            "flow_coverage_rate": {},
            "switch_utilization_rate": {},
            "total_execution_time": {},
            "congestion_rate": {},
            "system_efficiency": {},
            "end_to_end_latency": {}
        }
        
        for alg_name, metrics in results.items():
            metrics_data["flow_coverage_rate"][alg_name] = metrics.stage1_metrics.flow_coverage_rate
            metrics_data["switch_utilization_rate"][alg_name] = metrics.stage1_metrics.switch_utilization_rate
            metrics_data["total_execution_time"][alg_name] = metrics.total_execution_time
            metrics_data["congestion_rate"][alg_name] = metrics.stage2_metrics.congestion_rate
            metrics_data["system_efficiency"][alg_name] = metrics.system_efficiency
            metrics_data["end_to_end_latency"][alg_name] = metrics.end_to_end_latency
        
        comparison["metrics_comparison"] = metrics_data
        
        # 计算排名（越大越好的指标）
        for metric in ["flow_coverage_rate", "system_efficiency"]:
            sorted_algs = sorted(metrics_data[metric].items(), key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric] = [alg for alg, _ in sorted_algs]
            comparison["best_algorithm"][metric] = sorted_algs[0][0]
        
        # 计算排名（越小越好的指标）
        for metric in ["switch_utilization_rate", "total_execution_time", "congestion_rate", "end_to_end_latency"]:
            sorted_algs = sorted(metrics_data[metric].items(), key=lambda x: x[1])
            comparison["rankings"][metric] = [alg for alg, _ in sorted_algs]
            comparison["best_algorithm"][metric] = sorted_algs[0][0]
        
        # 生成总结
        comparison["summary"] = self._generate_comparison_summary(comparison)
        
        return comparison
    
    def _calculate_end_to_end_latency(self, stage1_result: Stage1Result, 
                                    stage2_result: Stage2Result, network: Network) -> float:
        """计算端到端延迟
        
        Args:
            stage1_result: 阶段一结果
            stage2_result: 阶段二结果
            network: 网络拓扑
            
        Returns:
            端到端延迟（毫秒）
        """
        total_latency = 0.0
        path_count = 0
        
        for measurement_node, paths in stage2_result.data_paths.items():
            for path in paths:
                if len(path) > 1:
                    latency = network.get_path_latency(path)
                    total_latency += latency
                    path_count += 1
        
        return total_latency / path_count if path_count > 0 else 0.0
    
    def _calculate_system_efficiency(self, stage1_metrics: Stage1Metrics, 
                                   stage2_metrics: Stage2Metrics) -> float:
        """计算系统效率指标
        
        Args:
            stage1_metrics: 阶段一指标
            stage2_metrics: 阶段二指标
            
        Returns:
            系统效率 (0-1)
        """
        # 综合考虑覆盖率、资源利用率和拥塞情况
        coverage_score = stage1_metrics.flow_coverage_rate
        efficiency_score = 1.0 - stage1_metrics.switch_utilization_rate  # 资源节约
        congestion_score = 1.0 - stage2_metrics.congestion_rate  # 低拥塞
        
        # 加权平均
        system_efficiency = (0.5 * coverage_score + 0.3 * efficiency_score + 0.2 * congestion_score)
        return max(0.0, min(1.0, system_efficiency))
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """生成对比分析总结
        
        Args:
            comparison: 对比分析数据
            
        Returns:
            总结信息
        """
        summary = {}
        
        # 找出在多个指标上表现最好的算法
        best_counts = defaultdict(int)
        for metric, best_alg in comparison["best_algorithm"].items():
            best_counts[best_alg] += 1
        
        overall_best = max(best_counts.items(), key=lambda x: x[1])
        summary["overall_best"] = f"{overall_best[0]} (在{overall_best[1]}个指标上最优)"
        
        # 各算法的特点总结
        for alg in comparison["algorithms"]:
            strengths = []
            for metric, best_alg in comparison["best_algorithm"].items():
                if best_alg == alg:
                    strengths.append(metric)
            
            if strengths:
                summary[f"{alg}_strengths"] = f"在{', '.join(strengths)}方面表现最佳"
            else:
                summary[f"{alg}_strengths"] = "无明显优势指标"
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息
        
        Returns:
            统计信息字典
        """
        if not self.evaluation_history:
            return {"message": "没有评估数据"}
        
        # 统计各算法组合的数量
        algorithm_counts = defaultdict(int)
        for metrics in self.evaluation_history:
            algorithm_counts[metrics.algorithm_combination] += 1
        
        # 计算平均性能
        avg_metrics = {
            "avg_flow_coverage": sum(m.stage1_metrics.flow_coverage_rate for m in self.evaluation_history) / len(self.evaluation_history),
            "avg_execution_time": sum(m.total_execution_time for m in self.evaluation_history) / len(self.evaluation_history),
            "avg_system_efficiency": sum(m.system_efficiency for m in self.evaluation_history) / len(self.evaluation_history),
        }
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "algorithm_counts": dict(algorithm_counts),
            "average_metrics": avg_metrics
        }

# 向后兼容的MonPlanEvaluator类
class MonPlanEvaluator(ComprehensiveEvaluator):
    """MonPlan评估器（向后兼容）"""
    
    def __init__(self):
        """初始化MonPlan评估器"""
        stage1_evaluator = Stage1Evaluator()
        stage2_evaluator = Stage2Evaluator()
        super().__init__(stage1_evaluator, stage2_evaluator)
    
    def evaluate_stage1(self, stage1_result: Stage1Result, flows: List[Flow], 
                       network: Network, algorithm_name: str = "Unknown") -> Stage1Metrics:
        """评估阶段一结果（向后兼容方法）"""
        return self.stage1_evaluator.evaluate(stage1_result, flows, network, algorithm_name)
    
    def evaluate_stage2(self, stage2_result: Stage2Result, 
                       algorithm_name: str = "Unknown") -> Stage2Metrics:
        """评估阶段二结果（向后兼容方法）"""
        return self.stage2_evaluator.evaluate(stage2_result, algorithm_name)

