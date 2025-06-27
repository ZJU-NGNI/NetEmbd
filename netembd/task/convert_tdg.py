import os
import csv

def convert_tdg_to_csv(tdg_file: str, output_dir: str):
    """将TDG配置文件转换为mats.csv和deps.csv
    
    Args:
        tdg_file: TDG配置文件路径
        output_dir: 输出目录路径
    """
    # 获取当前工作目录的绝对路径
    base_dir = os.path.abspath(os.getcwd())
    abs_output_dir = os.path.join(base_dir, output_dir)
    
    # 创建输出目录
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # 读取TDG文件
    with open(tdg_file, 'r') as f:
        lines = f.readlines()
    
    # 获取MAT数量
    mat_count = int(lines[0].strip().split()[-1])
    
    # 生成mats.csv
    mats_file = os.path.join(abs_output_dir, 'mats.csv')
    with open(mats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vnf_id', 'alu', 'stage', 'sram'])
        for i in range(mat_count):
            writer.writerow([i+1, 1, 1, 276480])
    
    # 生成deps.csv
    deps_file = os.path.join(abs_output_dir, 'deps.csv')
    with open(deps_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'data_size'])
        for line in lines[1:]:
            if line.strip():
                _, src, dst = line.strip().split()
                writer.writerow([int(src), int(dst), 1024])
                
def convert_all_tdgs(input_dir: str, output_base_dir: str):
    """转换所有TDG配置文件
    
    Args:
        input_dir: 输入目录路径
        output_base_dir: 输出基目录路径
    """
    # 获取当前工作目录的绝对路径
    base_dir = os.path.abspath(os.getcwd())
    abs_input_dir = os.path.join(base_dir, input_dir)
    abs_output_base_dir = os.path.join(base_dir, output_base_dir)
    
    # 处理sketch目录下的文件
    sketch_dir = os.path.join(abs_input_dir, 'sketch')
    if os.path.exists(sketch_dir):
        for file in os.listdir(sketch_dir):
            if file.endswith('.txt'):
                tdg_file = os.path.join(sketch_dir, file)
                output_dir = os.path.join(abs_output_base_dir, os.path.splitext(file)[0])
                convert_tdg_to_csv(tdg_file, output_dir)
    
    # 处理根目录下的merge文件
    for file in os.listdir(abs_input_dir):
        if file.startswith('merge_') and file.endswith('.txt'):
            tdg_file = os.path.join(abs_input_dir, file)
            output_dir = os.path.join(abs_output_base_dir, os.path.splitext(file)[0])
            convert_tdg_to_csv(tdg_file, output_dir)
            
if __name__ == '__main__':
    input_dir = os.path.join('src', 'input', 'program_tdg')
    output_dir = os.path.join('src_v3', 'examples')
    convert_all_tdgs(input_dir, output_dir)