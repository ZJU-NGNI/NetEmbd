"""任务DAG接口定义。

此模块定义了任务DAG的核心抽象接口，包括：
1. VNF管理：加载和查询VNF节点信息
2. 依赖管理：加载和查询节点间依赖关系
3. 资源需求：计算VNF的资源消耗
4. 通信开销：计算节点间的数据传输量

Typical usage example:

    from netembd.interfaces import BaseTask
    
    class CustomTask(BaseTask):
        def load_vnfs(self, vnf_csv: str) -> None:
            # 自定义VNF加载逻辑
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class VNFResource:
    """VNF资源需求数据类"""
    alu: int  # 需要的算术逻辑单元数量
    stage: int  # 需要的流水线阶段数
    sram: int  # 需要的SRAM容量(KB)

@dataclass
class VNFDependency:
    """VNF依赖关系数据类"""
    source: int  # 源VNF ID
    target: int  # 目标VNF ID
    data_size: int  # 数据传输量(Bytes)

class BaseTask(ABC):
    """任务DAG抽象基类"""
    
    @abstractmethod
    def load_vnfs(self, vnf_csv: str) -> None:
        """从CSV文件加载VNF配置
        
        Args:
            vnf_csv: VNF配置CSV文件路径，包含VNF ID和资源需求
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        pass
    
    @abstractmethod
    def load_dependencies(self, dep_csv: str) -> None:
        """从CSV文件加载依赖关系
        
        Args:
            dep_csv: 依赖关系CSV文件路径，包含源VNF、目标VNF和数据大小
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        pass
    
    @abstractmethod
    def get_vnfs(self) -> List[int]:
        """获取所有VNF ID
        
        Returns:
            VNF ID列表
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[VNFDependency]:
        """获取所有依赖关系
        
        Returns:
            依赖关系对象列表
        """
        pass
    
    @abstractmethod
    def get_vnf_resource(self, vnf_id: int) -> Optional[VNFResource]:
        """获取VNF的资源需求
        
        Args:
            vnf_id: VNF ID
        
        Returns:
            VNF资源需求对象，如果VNF不存在则返回None
        """
        pass
    
    @abstractmethod
    def get_dependency_data_size(self, source: int, target: int) -> Optional[int]:
        """获取两个VNF之间的数据传输量
        
        Args:
            source: 源VNF ID
            target: 目标VNF ID
        
        Returns:
            数据传输量（Bytes），如果依赖关系不存在则返回None
        """
        pass
    
    @abstractmethod
    def get_predecessors(self, vnf_id: int) -> List[int]:
        """获取VNF的所有前驱节点
        
        Args:
            vnf_id: VNF ID
        
        Returns:
            前驱VNF ID列表
        """
        pass
    
    @abstractmethod
    def get_successors(self, vnf_id: int) -> List[int]:
        """获取VNF的所有后继节点
        
        Args:
            vnf_id: VNF ID
        
        Returns:
            后继VNF ID列表
        """
        pass
    
    @abstractmethod
    def get_critical_path(self) -> List[int]:
        """计算关键路径
        
        Returns:
            关键路径上的VNF ID列表
        """
        pass