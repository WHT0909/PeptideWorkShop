import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from PeptideWorkShop.data import load_data_from_csv, turn_fasta_to_csv
from PeptideWorkShop.augmentation import add_gaussian_noise
from PeptideWorkShop.features import get_KL_diffEntropy

# # 1. 可选：将 fasta 转为 csv
# fasta_path = r'./demo_data/fasta_data/demo_fasta.fasta'
# turn_fasta_to_csv(fasta_path=fasta_path, out_csv_path=None)

# 2. 从 csv 中获取特征和标签
feature_data_path = './demo_data/feature_data/demo_csv.csv' # 已经提取好的特征
features, labels = load_data_from_csv(path=feature_data_path, feature_out_path=None, label_out_path=None)
print(features.shape, labels.shape)

# 3. 可选：高斯噪声增强
features = np.array(features)
labels = np.array(labels)
X_input = np.concatenate((features, labels), axis=1)
agn_features, agn_labels = add_gaussian_noise(X=X_input,
                                              mean=0.,
                                              std=0.01,
                                              n=10,
                                              random_state=42,
                                              out_path=None)
print(agn_features.shape, agn_labels.shape)

# 4. 可选：KL微分熵提取
KL_features, KL_labels = get_KL_diffEntropy(features=agn_features,
                                            labels=agn_labels,
                                            dim=2,
                                            k_start=3,
                                            save_path=None)
print(KL_features.shape)
print(KL_labels.shape)