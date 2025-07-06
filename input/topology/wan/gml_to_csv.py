import os
import re

def parse_gml_file(gml_file):
    """解析GML文件，提取节点和边的信息"""
    nodes = []
    edges = []
    current_section = None
    current_item = {}
    
    with open(gml_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line == 'node [':  # 开始节点部分
                current_section = 'node'
                current_item = {}
            elif line == 'edge [':  # 开始边部分
                current_section = 'edge'
                current_item = {}
            elif line == ']':  # 结束当前部分
                if current_section == 'node' and 'id' in current_item:
                    nodes.append(current_item)
                elif current_section == 'edge' and 'source' in current_item and 'target' in current_item:
                    edges.append(current_item)
                current_section = None
            elif current_section:
                # 解析键值对
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    # 移除引号
                    value = value.strip('"')
                    # 对于数字类型的值进行转换
                    if value.isdigit():
                        value = int(value)
                    current_item[key] = value
    
    return nodes, edges

def generate_csv_files(gml_file, output_dir):
    """生成nodes.csv和edges.csv文件"""
    nodes, edges = parse_gml_file(gml_file)
    
    # 生成nodes.csv
    with open(os.path.join(output_dir, 'nodes.csv'), 'w', newline='') as f:
        f.write('node_id,alu,stage,sram,pipeline\n')
        for node in nodes:
            # 所有节点使用相同的硬件配置
            f.write(f"{node['id']},16,4,276480,2\n")
    
    # 生成edges.csv
    with open(os.path.join(output_dir, 'edges.csv'), 'w', newline='') as f:
        f.write('source,target,bandwidth,latency\n')
        for edge in edges:
            # 带宽固定为10000，延迟随机在20-50ms之间
            import random
            latency = random.randint(20, 50)
            f.write(f"{edge['source']},{edge['target']},10000,{latency}\n")

def process_all_topologies(base_dir):
    """处理所有拓扑文件"""
    # 遍历所有子目录
    for topology_dir in os.listdir(base_dir):
        topology_path = os.path.join(base_dir, topology_dir)
        if not os.path.isdir(topology_path):
            continue
            
        # 查找.gml.txt文件
        gml_files = [f for f in os.listdir(topology_path) if f.endswith('.gml.txt')]
        if not gml_files:
            continue
            
        # 处理找到的GML文件
        gml_file = os.path.join(topology_path, gml_files[0])
        print(f'Processing {gml_file}...')
        generate_csv_files(gml_file, topology_path)
        print(f"Generated CSV files in {topology_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    process_all_topologies(base_dir)