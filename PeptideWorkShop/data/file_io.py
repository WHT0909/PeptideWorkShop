import os
import numpy as np
import pandas as pd
from pandas import DataFrame

def check_path(out_path):
    """
    检查输出文件路径是否合法，并保证存在上级目录
    :param out_path: 文献输出路径
    """
    if out_path is not None:
        dir_name = os.path.dirname(out_path)
        dir_abs_path = os.path.abspath(dir_name)
        if not os.path.exists(dir_abs_path):
            os.mkdir(dir_abs_path)

def load_data_from_csv(path, feature_out_path=None, label_out_path=None):
    """
    从 csv 文件中获取 features 和 labels
    csv 格式为 第一列名称 第二列标签 第三列到最后一列为特征
    :param path: csv 文件路径
    :param feature_out_path: 将特征输出为 csv 文件，可选
    :param label_out_path: 将标签输出为 csv 文件，可选
    :return: features, labels 均为 numpy 数组
    """
    data = pd.read_csv(path)
    if data.shape[1] <= 2:
        raise ValueError("错误：请检查输入数据的列数")
    data_list = data.values
    labels = data_list[:, 2].astype(int).reshape((-1, 1))
    features = data_list[:, 2:-1].astype(float)
    if feature_out_path is not None:
        feature_out_csv_data = np.hstack((data_list[:, 0].reshape((-1, 1)), features))
        check_path(feature_out_path)
        df = DataFrame(feature_out_csv_data)
        df.to_csv(feature_out_path, header=True, index=False)
    if label_out_path is not None:
        label_out_csv_data = np.hstack((data_list[:, 0].reshape((-1, 1)), labels))
        check_path(label_out_path)
        df = DataFrame(label_out_csv_data)
        df.to_csv(label_out_path, header=True, index=False)
    return features, labels

def turn_fasta_to_csv(fasta_path, out_csv_path):
    """
    将输入的 fasta 序列转换为 csv, 输入 fasta 格式参照 'demo/fasta_data/old_demo_fasta.fasta'
    :param fasta_path: fasta 文件路径
    :param out_csv_path:输出文件路径
    :return: csv，第一列为 name 第二列为 label
    """
    # 检查文件名
    if fasta_path.split('.')[-1] not in ('fasta', 'txt', 'fa'):
        raise ValueError("错误：请检查输入的 fasta 文件后缀")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError("错误：该文件似乎不存在")
    # 检查输出路径
    dir_name = os.path.dirname(out_csv_path)
    dir_path = os.path.abspath(dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    csv_rows = []
    with open(fasta_path, 'r', encoding='utf-8') as file:
        lines = list(file)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('>') or ('|' in line):
                # 处理标题行
                line_clean = line.lstrip('>')
                parts = line_clean.split('|')
                if len(parts) < 2:
                    raise ValueError("标题行格式错误，需要以'|'分隔")
                pep_name = parts[0].strip()
                pep_label = parts[-1].strip()
                pep_sequence = ''
                i += 1
                while i < len(lines):
                    # 处理跨行序列
                    seq_line = lines[i].strip()
                    if seq_line.startswith('>') or ('|' in seq_line):
                        break
                    if seq_line:
                        pep_sequence += seq_line
                    i += 1
                csv_rows.append([pep_name, pep_sequence, pep_label])
            else: # 处理空行
                i += 1
    # print(csv_rows)
    if not csv_rows:
        raise RuntimeError("没有从 fasta 中提取到序列")
    df = DataFrame(csv_rows, columns=['name', 'sequence', 'label'])
    out_abs_path = os.path.abspath(out_csv_path)
    df.to_csv(out_abs_path, index=True, encoding='utf-8')
    print(f"处理完成, csv 文件已保存至{out_abs_path}")

def main():
    load_data_from_csv(path='../../examples/demo_data/feature_data/demo_csv.csv',
                       label_out_path='./test/example_label_out_csv.csv',
                       feature_out_path='./test/example_feature_out_csv.csv')

    # turn_fasta_to_csv(fasta_path='../../examples/demo_data/fasta_data/demo_fasta.fasta',
    #                   out_csv_path='./test/example_turn_fasta_to_csv.csv')
    return

if __name__ == '__main__':
    main()

