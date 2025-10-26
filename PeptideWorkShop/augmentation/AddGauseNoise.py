import numpy as np
import pandas as pd
import sys
from pathlib import Path
from pandas import DataFrame
from PeptideWorkShop.data import load_data_from_csv, check_path


def AddGauseNoise(X, mean=0., std=0.01, n=10, random_state=42, out_path=None):
    """
    高斯噪声增强特征，向特征中添加噪声扰动
    :param X: 形状为 (m, n) 的 numpy 矩阵, m为样本数, n的最后一列为 label, 前面是 feature
    :param mean: 高斯噪声的均值
    :param std: 高斯噪声的标准差
    :param n: 每个样本点生成的样本数
    :param random_state: 随机种子
    :return: 分别返回增强后的 features 和 labels
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("错误：输入的矩阵不是 numpy 数组")
    if X.shape[1] < 2:
        raise ValueError("错误：矩阵维度出错。请检查输入的矩阵是否包含了标签列")
    if random_state:
        np.random.seed(random_state)
    features = X[:, :-1].astype(float)
    labels = X[:, -1].astype(float).reshape((-1, 1))
    agn_features, agn_labels = [], []
    for feature, label in zip(features, labels):
        for _ in range(n):
            noise = np.random.normal(mean, std, size=feature.shape)
            agn_features.append(feature + noise)
            agn_labels.append(label)
    agn_features = np.array(agn_features)
    agn_labels = np.array(agn_labels)
    if out_path:
        check_path(out_path)
        agn_data = np.concatenate((agn_labels, agn_features), axis=1)
        df = DataFrame(agn_data)
        df.to_csv(out_path, encoding='utf-8', index=True)

    return agn_features, agn_labels

def main():
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent.parent
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))
    features, labels = load_data_from_csv(path='../../examples/demo_data/feature_data/demo_csv.csv')
    # print(features.shape, labels.shape)
    total_data = np.concatenate((features, labels), axis=1)
    AddGauseNoise(total_data, out_path='./test/agn_data.csv')

if __name__ == '__main__':
    main()