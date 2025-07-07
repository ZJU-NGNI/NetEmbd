"""网络拓扑接口定义。

此模块定义了网络拓扑的核心抽象接口，包括：
1. 节点管理：加载和查询节点信息
2. 边管理：加载和查询边信息
3. 资源管理：节点资源分配和释放
4. 路径计算：最短路径和延迟计算

Typical usage example:

    from netembd.interfaces import BaseNetwork
    
    class CustomNetwork(BaseNetwork):
        def load_nodes(self, node_csv: str) -> None:
            # 自定义节点加载逻辑
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 前向声明，避免循环导入
if False:
    from netembd.network.flow_generator import Flow

@dataclass
class NodeResource:
    """节点资源数据类"""
    alu: int  # 算术逻辑单元数量
    stage: int  # 流水线阶段数
    sram: int  # SRAM容量(KB)
    programmable: Optional[bool] = False  # 是否可编程，默认为False
    control_node: Optional[bool] = False  # 是否为控制节点，默认为False

class BaseNetwork(ABC):
    """网络拓扑抽象基类"""
    
    @abstractmethod
    def load_nodes(self, node_csv: str) -> None:
        """从CSV文件加载节点配置
        
        Args:
            node_csv: 节点配置CSV文件路径，包含节点ID和资源信息
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        pass
    
    @abstractmethod
    def load_edges(self, edge_csv: str) -> None:
        """从CSV文件加载边配置
        
        Args:
            edge_csv: 边配置CSV文件路径，包含源节点、目标节点、带宽和延迟
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        pass
    
    @abstractmethod
    def get_nodes(self) -> List[int]:
        """获取所有节点ID
        
        Returns:
            节点ID列表
        """
        pass
    
    @abstractmethod
    def get_edges(self) -> List[Tuple[int, int]]:
        """获取所有边
        
        Returns:
            边列表，每条边为(源节点ID, 目标节点ID)元组
        """
        pass
    
    @abstractmethod
    def get_links(self) -> List['Link']:
        """获取所有链路对象
        
        Returns:
            链路对象列表，每个链路包含源节点、目标节点、带宽和延迟信息
        """
        pass
    
    @abstractmethod
    def get_node_resource(self, node_id: int) -> Optional[NodeResource]:
        """获取节点的资源信息
        
        Args:
            node_id: 节点ID
        
        Returns:
            节点资源对象，如果节点不存在则返回None
        """
        pass
    
    @abstractmethod
    def get_edge_latency(self, source: int, target: int) -> Optional[float]:
        """获取边的通信延迟
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            通信延迟（毫秒），如果边不存在则返回None
        """
        pass
    
    @abstractmethod
    def get_edge_bandwidth(self, source: int, target: int) -> Optional[float]:
        """获取边的带宽
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            带宽（Mbps），如果边不存在则返回None
        """
        pass
    
    @abstractmethod
    def get_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """计算两个节点间的最短路径
        
        Args:
            source: 源节点ID
            target: 目标节点ID
        
        Returns:
            节点ID列表表示的路径，如果不存在路径则返回None
        """
        pass
    
    @abstractmethod
    def get_path_latency(self, path: List[int]) -> float:
        """计算路径的总延迟
        
        Args:
            path: 节点ID列表表示的路径
        
        Returns:
            路径总延迟（毫秒）
        
        Raises:
            ValueError: 路径无效
        """
        pass
    
    @abstractmethod
    def generate_flows(self, num_flows: int, flow_size_range: Tuple[float, float] = (1.0, 1000.0), 
                      large_flow_threshold: float = 100.0) -> List['Flow']:
        """生成适合该网络类型的流
        
        Args:
            num_flows: 生成的流数量
            flow_size_range: 流大小范围
            large_flow_threshold: 大流阈值，超过此值使用sketch测量
        
        Returns:
            生成的流列表
        """
        pass