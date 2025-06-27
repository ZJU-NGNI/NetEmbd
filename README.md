# NetEmbd - 网络功能部署优化框架

NetEmbd是一个通用的网络功能部署优化框架，它能够将虚拟网络功能(VNF e.g. MATs)高效地部署到物理网络节点上。框架支持多种部署策略，包括启发式算法和精确求解方法。

## 功能特点

- 支持从CSV文件加载网络拓扑和任务DAG
- 提供统一的优化器接口，支持多种部署算法
- 自动处理节点资源约束和通信带宽限制
- 提供命令行接口和Python API
- 包含完整的单元测试和类型注解

## 安装

### 依赖要求

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- networkx >= 2.6.0
- gurobipy >= 9.5.0
- matplotlib >= 3.4.0

### 安装步骤

1. 克隆代码仓库：
```bash
git clone ...
cd netembd
```

2. 创建虚拟环境（可选但推荐）：
```bash
conda create -m netembd python=3.8
```

3. 安装依赖：
```bash
pip install -e .
```

4. 安装开发依赖（可选）：
```bash
pip install -e ".[dev]"
```

## 使用方法

### 命令行接口

```bash
# 使用启发式算法求解
python -m netembd optimize \
    --network-nodes nodes.csv \
    --network-edges edges.csv \
    --task-vnfs vnfs.csv \
    --task-deps deps.csv \
    --algorithm speed_heuristic \
    --output solution.csv

# 使用精确求解
python -m netembd optimize \
    --network-nodes nodes.csv \
    --network-edges edges.csv \
    --task-vnfs vnfs.csv \
    --task-deps deps.csv \
    --algorithm speed_exact \
    --time-limit 3600 \
    --output solution.csv
```

### Python API

```python
from netembd.network import Network
from netembd.task import Task
from netembd.optimize.speed import SpeedExact
from netembd.interfaces.base_optimizer import OptimizerConfig

# 加载网络拓扑
network = Network()
network.load_nodes("nodes.csv")
network.load_edges("edges.csv")

# 加载任务DAG
task = Task()
task.load_vnfs("vnfs.csv")
task.load_dependencies("deps.csv")

# 配置并运行优化算法
config = OptimizerConfig(
    time_limit=3600,  # 1小时时间限制
    gap_limit=0.01,   # 1%的gap限制
    verbose=False     # 不输出求解过程
)
optimizer = SpeedExact(network, task, config)
deployment = optimizer.solve()

# 保存结果
if deployment:
    deployment.save_to_csv("solution.csv")
    print(f"总通信延迟：{deployment.calculate_total_latency():.2f}ms")
```

## 输入文件格式

### 网络节点配置（nodes.csv）
```csv
node_id,alu,stage,sram
0,8,4,1024
1,16,8,2048
```

### 网络边配置（edges.csv）
```csv
source,target,bandwidth,latency
0,1,10000,1
1,2,10000,1
```

### VNF配置（vnfs.csv）
```csv
vnf_id,alu,stage,sram
0,2,1,256
1,4,2,512
```

### 依赖关系配置（deps.csv）
```csv
source,target,data_size
0,1,1024
1,2,2048
```

## 项目结构

```
netembd/
├── __init__.py
├── __main__.py           # 命令行入口
├── network/              # 网络拓扑相关
│   ├── __init__.py
│   └── network.py
├── task/                 # 任务DAG相关
│   ├── __init__.py
│   └── task.py
├── optimize/             # 优化算法相关
│   ├── __init__.py
│   ├── exact.py          # 基础精确求解
│   ├── heuristic.py      # 基础启发式算法
│   ├── speed/            # Speed策略优化器
│   │   ├── SpeedExact.py
│   │   └── SpeedHeuristic.py
│   └── hermes/           # Hermes策略优化器
│       ├── HermesHeuristic.py
│       └── __init__.py
├── deployment/           # 部署方案相关
│   ├── __init__.py
│   └── deployment.py
└── interfaces/           # 接口定义
    ├── __init__.py
    ├── base_network.py
    ├── base_task.py
    ├── base_deployment.py
    └── base_optimizer.py
```

## 开发指南

### 代码风格


运行代码质量检查：
```bash
None for now
```

### 运行测试
### 运行测试


```bash
查看 [evaluation](./netembd/evaluation) 目录
```

## 贡献指南

1. Fork本仓库
2. 创建功能分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 提交Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

如有问题或建议，请提交Issue或发送邮件至：chenzedi@zju.edu.cn