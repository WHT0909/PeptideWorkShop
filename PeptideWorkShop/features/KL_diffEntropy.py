import warnings
import numpy as np
import numpy.linalg as la
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree
import pandas as pd
import os


def get_KL_diffEntropy(features, labels, dim, save_path=None, k_start=3):
    """
    从numpy数组中提取微分熵，并保存结果（可选）
    :param features: np.ndarray - 特征数组，形状为(n_samples, n_features)
    :param labels: np.ndarray - 标签数组，形状为(n_samples,) 或 (n_samples, 1)
    :param dim: int - 要计算的k值数量（k从k_start开始，共dim个值，如dim=10则k=3~12）
    :param save_path: str - 结果保存的路径（可选，为None时不保存）
    :param k_start: int - 计算微分熵的起始k值，默认从3开始
    :return: tuple - (KL_features, KL_labels)，其中KL_features为微分熵特征数组，KL_labels为标签数组
    """

    # ---------------------- 1. 内部工具函数 ----------------------
    def add_noise(x, intens=1e-10):
        return x + intens * np.random.random_sample(x.shape)

    def query_neighbors(tree, x, k):
        return tree.query(x, k=k + 1)[0][:, k]

    def build_tree_with_metric(points, metric="chebyshev"):
        if points.shape[1] >= 20:
            return BallTree(points, metric=metric)
        return KDTree(points, metric=metric)

    def entropy_with_metrics(x, k=3, base=2, metric="chebyshev"):
        assert k <= len(x) - 1, f"k={k}需小于样本数-1（当前样本数{len(x)}）"
        x = np.asarray(x)
        n_elements, n_features = x.shape
        x = add_noise(x)
        tree = build_tree_with_metric(x, metric=metric)
        nn = query_neighbors(tree, x, k)
        const = digamma(n_elements) - digamma(k) + n_features * log(2)
        return (const + n_features * np.log(nn).mean()) / log(base)

    # ---------------------- 2. 数据校验与初始化 ----------------------
    # 确保features和labels形状匹配
    assert len(features) == len(labels), "features和labels的样本数必须一致"
    sample_num = len(features)
    feature_dim = features.shape[1] if len(features.shape) > 1 else 1
    print(f"成功接收数据：共{sample_num}个样本，每个样本{feature_dim}个特征")

    # 初始化结果列表（存储每个k值的微分熵）
    kl_feature_list = []

    # 确定要计算的k值列表
    k_values = [k_start + i for i in range(dim)]
    print(f"将计算的k值：{k_values}（共{dim}个）")

    # ---------------------- 3. 逐k值计算微分熵 ----------------------
    for set_k in k_values:
        print(f"正在计算 k={set_k} 的微分熵...")
        entropy_list = []

        # 逐样本计算当前k值的微分熵
        for i in range(sample_num):
            # 提取单个样本的特征，reshape为(特征数, 1)的二维数组
            sample_features = features[i].reshape(-1, 1)
            # 计算微分熵
            ent = entropy_with_metrics(sample_features, k=set_k)
            entropy_list.append(ent)

        # 将当前k值的熵结果加入列表
        kl_feature_list.append(entropy_list)

    # 将微分熵特征列表转换为数组（形状：(n_samples, dim)）
    KL_features = np.array(kl_feature_list).T

    # 确保标签数组形状统一（转为一维数组）
    KL_labels = labels.flatten()

    # ---------------------- 4. 结果保存（可选）----------------------
    if save_path:
        # 拼接特征和标签为DataFrame保存
        result_df = pd.DataFrame(KL_features, columns=[f'diffEntropy_k{k}' for k in k_values])
        result_df.insert(0, 'label', KL_labels)  # 标签插入第一列
        result_df.to_csv(save_path, index=False)
        print(f"微分熵计算完成！结果已保存至：{save_path}")

    return KL_features, KL_labels


# ---------------------- 示例调用 ----------------------
if __name__ == '__main__':
    # 生成示例数据（10个样本，每个样本5个特征）
    np.random.seed(42)
    features = np.random.randn(10, 5)  # 特征数组：(10, 5)
    labels = np.random.randint(0, 2, size=10)  # 标签数组：(10,)

    # 计算微分熵
    kl_feats, kl_labels = get_KL_diffEntropy(
        features=features,
        labels=labels,
        dim=2,  # 计算2个k值（k=3,4）
        save_path="./test_KL.npy.csv",  # 可选保存路径
        k_start=3
    )

    print("\n微分熵特征形状：", kl_feats.shape)  # 应为(10, 2)
    print("标签形状：", kl_labels.shape)        # 应为(10,)
