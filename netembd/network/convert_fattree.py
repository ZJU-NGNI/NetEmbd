import os
import csv

def convert_fattree_to_csv(output_dir: str):
    """将FatTree拓扑转换为edges.csv和nodes.csv
    
    Args:
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # FatTree拓扑边的定义
    edges = [
        (0, 4), (0, 6), (0, 8), (0, 10),
        (1, 4), (1, 6), (1, 8), (1, 10),
        (2, 5), (2, 7), (2, 9), (2, 11),
        (3, 5), (3, 7), (3, 9), (3, 11),
        (4, 12), (4, 13), (5, 12), (5, 13),
        (6, 14), (6, 15), (7, 14), (7, 15),
        (8, 16), (8, 17), (9, 16), (9, 17),
        (10, 18), (10, 19), (11, 18), (11, 19)
    ]
    
    # 生成edges.csv
    edges_file = os.path.join(output_dir, 'edges.csv')
    with open(edges_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'bandwidth', 'latency'])
        for src, dst in edges:
            # 设置带宽和延迟，参考示例格式
            writer.writerow([src, dst, 10000, 40])
    
    # 生成nodes.csv
    nodes_file = os.path.join(output_dir, 'nodes.csv')
    with open(nodes_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'alu', 'stage', 'sram', 'pipeline'])
        # 生成20个节点，参考示例格式设置资源
        for i in range(20):
            writer.writerow([i, 16, 4, 2048, 2])
            
if __name__ == '__main__':
    output_dir = os.path.join('src_v3', 'examples', 'FatTree_20')
    convert_fattree_to_csv(output_dir)