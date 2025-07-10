import os
from netembd.network import Network

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 从文件加载网络
network = Network()
network.load_nodes(os.path.join(current_dir, "nodes.csv"))
network.load_edges(os.path.join(current_dir, "edges.csv"))

# 可视化网络
network.visualize()