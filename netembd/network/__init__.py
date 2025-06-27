"""网络拓扑管理模块。

此模块提供了网络拓扑的核心功能，包括：
1. 从CSV文件加载网络结构
2. 计算最短路径和延迟矩阵
3. 管理节点资源分配
4. 提供网络属性查询接口

Typical usage example:

    from netembd.network import Network
    
    network = Network()
    network.load_nodes("nodes.csv")
    network.load_edges("edges.csv")
    path = network.get_shortest_path(source=0, target=5)
"""

from .network import Network

__all__ = ["Network"]