"""部署方案管理模块。

此模块提供了部署方案的核心功能，包括：
1. 管理VNF到物理节点的映射关系
2. 计算部署方案的资源使用情况
3. 验证部署方案的可行性
4. 提供部署方案的导入导出功能

Typical usage example:

    from netembd.deployment import Deployment
    
    deployment = Deployment(network, task)
    deployment.assign_vnf(vnf_id=0, node_id=1)
    deployment.save_to_csv("deployment.csv")
    latency = deployment.calculate_total_latency()
"""

from .deployment import Deployment

__all__ = ["Deployment"]