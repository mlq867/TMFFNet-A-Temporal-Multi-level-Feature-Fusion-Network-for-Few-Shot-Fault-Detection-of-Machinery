import torch
from torch.utils.data import Dataset
import random

class FewShotDataset(Dataset):
    """
    Few-shot episode dataset for N-way K-shot training.

    Args:
        data: Tensor of shape [num_samples, seq_len, feature_dim]
        labels: Tensor of shape [num_samples], with class labels
        episode_num: Number of episodes per epoch
        N: Number of classes per episode (N-way)
        K: Number of support samples per class (K-shot)
        Q: Number of query samples per class
    """
    def __init__(self, data, labels, episode_num=1000, N=5, K=1, Q=1):
        super(FewShotDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.episode_num = episode_num
        self.N = N
        self.K = K
        self.Q = Q

        # 构建每类样本索引，方便采样
        self.class_to_indices = {}
        classes = torch.unique(labels)
        for c in classes:
            self.class_to_indices[c.item()] = torch.nonzero(labels == c).flatten().tolist()

    def __len__(self):
        return self.episode_num

    def __getitem__(self, idx):
        """
        返回一个 episode：
        - 支持集: [N*K, seq_len, feature_dim]
        - 支持集标签: [N*K]
        - 查询集: [N*Q, seq_len, feature_dim]
        - 查询集标签: [N*Q]
        """
        # 随机选择 N 类
        sampled_classes = random.sample(list(self.class_to_indices.keys()), self.N)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for i, c in enumerate(sampled_classes):
            indices = self.class_to_indices[c].copy()
            random.shuffle(indices)

            # 取 K 个作为支持集
            support_indices = indices[:self.K]
            support_x.append(self.data[support_indices])
            support_y.append(torch.full((self.K,), i))  # 统一映射为 0~N-1 类

            # 剩下的取 Q 个作为查询集
            query_indices = indices[self.K:self.K+self.Q]
            query_x.append(self.data[query_indices])
            query_y.append(torch.full((self.Q,), i))

        # 拼接
        support_x = torch.cat(support_x, dim=0)  # [N*K, seq_len, feature_dim]
        support_y = torch.cat(support_y, dim=0)  # [N*K]
        query_x = torch.cat(query_x, dim=0)      # [N*Q, seq_len, feature_dim]
        query_y = torch.cat(query_y, dim=0)      # [N*Q]

        return support_x, support_y, query_x, query_y
