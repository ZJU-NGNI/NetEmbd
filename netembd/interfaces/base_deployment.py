"""部署方案接口定义。

此模块定义了部署方案的核心抽象接口，包括：
1. 部署管理：分配VNF到物理节点
2. 资源管理：跟踪和验证资源使用情况
3. 性能评估：计算总通信延迟等指标
4. 结果导出：保存和可视化部署方案

Typical usage example:

    from netembd.interfaces import BaseDeployment
    
    class CustomDeployment(BaseDeployment):
        def assign_vnf(self, vnf_id: int, node_id: int) -> None:
            # 自定义VNF分配逻辑
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .base_network import BaseNetwork
from .base_task import BaseTask

@dataclass
class Assignment:
    """VNF分配信息数据类"""
    vnf_id: int  # VNF ID
    node_id: int  # 物理节点ID
    stage_id: Optional[int] = None  # 流水线阶段ID（可选）

class BaseDeployment(ABC):
    """部署方案抽象基类"""
    
    @abstractmethod
    def assign_vnf(self, vnf_id: int, node_id: int, stage_id: Optional[int] = None) -> None:
        """将VNF分配到物理节点
        
        Args:
            vnf_id: VNF ID
            node_id: 物理节点ID
            stage_id: 流水线阶段ID（可选）
        
        Raises:
            ValueError: 分配无效（如资源不足）
        """
        pass
    
    @abstractmethod
    def get_vnf_assignment(self, vnf_id: int) -> Optional[Assignment]:
        """获取VNF的分配信息
        
        Args:
            vnf_id: VNF ID
        
        Returns:
            分配信息对象，如果VNF未分配则返回None
        """
        pass
    
    @abstractmethod
    def get_node_vnfs(self, node_id: int) -> Set[int]:
        """获取分配到节点的所有VNF
        
        Args:
            node_id: 物理节点ID
        
        Returns:
            VNF ID集合
        """
        pass
    
    @abstractmethod
    def calculate_node_resource_usage(self, node_id: int) -> Dict[str, float]:
        """计算节点的资源使用情况
        
        Args:
            node_id: 物理节点ID
        
        Returns:
            资源使用情况字典，包含'alu'、'stage'和'sram'的使用率
        """
        pass
    
    @abstractmethod
    def calculate_total_latency(self) -> float:
        """计算总通信延迟
        
        Returns:
            总通信延迟（毫秒）
        """
        pass
    
    @abstractmethod
    def is_feasible(self) -> bool:
        """检查部署方案是否可行
        
        Returns:
            如果方案满足所有约束则返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def save_to_csv(self, file_path: str) -> None:
        """将部署方案保存为CSV文件
        
        Args:
            file_path: 输出文件路径
        
        Raises:
            IOError: 写入文件失败
        """
        pass
    
    @abstractmethod
    def visualize(self, output_path: Optional[str] = None) -> None:
        """可视化部署方案
        
        Args:
            output_path: 输出文件路径（可选），如果不指定则显示图形
        """
        pass