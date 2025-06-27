"""Hermes启发式算法模块。

此模块实现了基于任务分层和资源感知的Hermes启发式算法，用于解决VNF部署优化问题。

Typical usage example:

    from netembd.optimize.hermes import HermesHeuristic
    
    optimizer = HermesHeuristic(network, task)
    deployment = optimizer.solve()
"""

from netembd.optimize.hermes.HermesHeuristic import HermesHeuristic

__all__ = ['HermesHeuristic']