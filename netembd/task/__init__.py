"""任务DAG管理模块。

此模块提供了任务DAG的核心功能，包括：
1. 从CSV文件加载VNF和依赖关系
2. 计算关键路径和资源需求
3. 管理VNF之间的依赖关系
4. 提供任务属性查询接口

Typical usage example:

    from netembd.task import Task
    
    task = Task()
    task.load_vnfs("vnfs.csv")
    task.load_dependencies("deps.csv")
    critical_path = task.get_critical_path()
"""

from .task import Task

__all__ = ["Task"]