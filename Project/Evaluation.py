
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from Functions.DataLoader import Dataset
from Functions.normTranDataLoader import FewShotDataset
from typing import Optional, Dict, Tuple

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

def confidence_interval(data, confidence: float = 0.95, z_threshold: float = 1.0) -> Tuple[float, float, int]:
    """
    计算给定序列的置信区间（默认95%），并自动排除异常值

    参数:
        data (np.ndarray): 数据序列，例如多次实验的准确率
        confidence (float): 置信水平，默认0.95（即95%置信区间）
        z_threshold (float): Z-score阈值，超过则视为异常点（默认3.0）

    返回:
        mean (float): 样本均值（剔除异常值后）
        ci (float): 置信区间半宽（mean ± ci）
    """
    data = np.array(data, dtype=float)

    # ---- 异常值剔除 ----
    z_scores = np.abs((data - np.mean(data)) / np.std(data, ddof=1))
    filtered_data = data[z_scores < z_threshold]

    if len(filtered_data) < 2:
        raise ValueError("有效数据过少，无法计算置信区间。")

    # ---- 置信区间计算 ----
    mean = np.mean(filtered_data)
    std = np.std(filtered_data, ddof=1)
    n = len(filtered_data)
    z = 1.96 if confidence == 0.95 else 1.645  # 近似正态分布
    ci = z * std / np.sqrt(n)
    return mean, ci, filtered_data.shape[0]


def get_dataset_config(dataset_name: str) -> Dict:
    """根据数据集名称返回评估所需的参数"""
    if dataset_name == 'pumpJSU':
        N = 5
        return dict(
            N=N, K=1, Q=1, episode_num=1,
            data_path=r"./Dataset/pumpJD/",
            numClass=N,
            numPerClassTrain=10,
            numPerClassTest=400
        )
    elif dataset_name == 'CRB':
        N = 13
        return dict(
            N=N, K=1, Q=1, episode_num=1,
            data_path=r"./Dataset/CRB/",
            numClass=N,
            numPerClassTrain=10,
            numPerClassTest=50
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_data(cfg, dataset_name: str):
    """加载数据集，返回训练和测试 Tensor"""
    np.random.seed(42)
    dataset = Dataset(path=cfg['data_path'], datasetName=dataset_name, mode=cfg['mode'],
                      numPerClassTrain=cfg['numPerClassTrain'], numPerClassTest=cfg['numPerClassTest'],
                      frame_len=cfg['frame_len'], num_frames=cfg['num_frames'])
    X_train = torch.tensor(dataset.X_train, dtype=torch.float32)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long)
    X_test = torch.tensor(dataset.X_test, dtype=torch.float32)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long)
    return X_train, y_train, X_test, y_test, cfg


def create_test_loader(X_test, y_test, cfg, Q_override: Optional[int] = None) -> DataLoader:
    """构建 FewShotDataset 测试 DataLoader"""
    Q = Q_override if Q_override is not None else cfg['Q']
    test_episodes = FewShotDataset(X_test, y_test, episode_num=cfg['episode_num'],
                                   N=cfg['N'], K=cfg['K'], Q=Q)
    return DataLoader(test_episodes, batch_size=1, shuffle=False)


def data_prepare(cfg: dict,  dataset_name: str, Q_override: Optional[int] = None):
    X_train, y_train, X_test, y_test, cfg = load_data(cfg, dataset_name)
    print(f"Loaded dataset '{dataset_name}': X_train={X_train.shape}, X_test={X_test.shape}")
    test_loader = create_test_loader(X_test, y_test, cfg, Q_override)
    return test_loader


def evaluate_model(model_file: str, num_repeats: int = 200, test_loader: Optional[DataLoader] = None):
    """评估指定模型在指定数据集上的性能"""
    model = torch.load(model_file).to(device)
    model.eval()
    acc_list = []
    if test_loader is None:
        print("test_loader is None")
        return
    with torch.no_grad():
        for _ in range(num_repeats):
            total_acc = 0.0
            for support_x, support_y, query_x, query_y in test_loader:
                support_x = support_x.squeeze(0).transpose(1, 2).to(device)
                support_y = support_y.squeeze(0).to(device)
                query_x = query_x.squeeze(0).transpose(1, 2).to(device)
                query_y = query_y.squeeze(0).to(device)

                output = model(query_x, support_x, support_y)
                logits = output[0] if isinstance(output, tuple) else output

                preds = torch.argmax(logits, dim=1)
                acc = (preds == query_y).float().mean()
                total_acc += acc.item()
            acc_list.append(total_acc)
    # 置信区间计算函数（确保你已有定义 confidence_interval）
    mean, ci, num = confidence_interval(acc_list, z_threshold=1.0)
    print(f"Total Accuracy ({num} runs) = {mean*100:.2f}% ± {ci*100:.2f}%")

import torch

def extract_features_batch(model, x):
    """
    一次 forward，抽取三路特征。
    x: [B, T, C_in]
    return: x_trans_pooled, fusion_pooled, pooled
    """
    with torch.no_grad():
        # Transformer 分支
        x_trans = model.trans(x.transpose(1, 2)).transpose(1, 2)
        x_trans = model.conv1(x_trans)
        x_trans = model.bn1(x_trans)
        x_trans = model.relu(x_trans)
        x_trans_pooled = model.global_pool(x_trans).squeeze(-1)

        # 多层残差融合分支
        x1, x2, x3, x4 = model.resNet(x)
        F1, F2, F3, F4 = model.align(x1, x2, x3, x4)
        fusion_feat = model.fusion([F1, F2, F3, F4])
        fusion_pooled = model.global_pool(fusion_feat).squeeze(-1)

        # 拼接
        pooled = torch.cat([x_trans_pooled, fusion_pooled], dim=1)

    return x_trans_pooled, fusion_pooled, pooled


def compute_prototypes(feats, labels):
    classes = torch.unique(labels)
    proto = []
    for c in classes:
        proto.append(feats[labels == c].mean(dim=0))
    return torch.stack(proto)


def evaluate_model_all(model_file, test_loader, num_repeats=200):
    model = torch.load(model_file).to(device)
    model.eval()
    acc_x_trans, acc_fusion, acc_pooled = [], [], []

    with torch.no_grad():
        for _ in range(num_repeats):
            total_x, total_f, total_p = 0.0, 0.0, 0.0

            for support_x, support_y, query_x, query_y in test_loader:
                support_x = support_x.squeeze(0).transpose(1, 2).to(device)
                support_y = support_y.squeeze(0).to(device)
                query_x = query_x.squeeze(0).transpose(1, 2).to(device)
                query_y = query_y.squeeze(0).to(device)

                # ===== 一次 forward 提取三路特征 =====
                all_x = torch.cat([support_x, query_x], dim=0)
                x_trans_all, fusion_all, pooled_all = extract_features_batch(model, all_x)

                B_support = support_x.size(0)
                support_x_trans = x_trans_all[:B_support]
                query_x_trans = x_trans_all[B_support:]
                support_fusion = fusion_all[:B_support]
                query_fusion = fusion_all[B_support:]
                support_pooled = pooled_all[:B_support]
                query_pooled = pooled_all[B_support:]

                # ===== 计算 prototypes =====
                proto_x = compute_prototypes(support_x_trans, support_y)
                proto_f = compute_prototypes(support_fusion, support_y)
                proto_p = compute_prototypes(support_pooled, support_y)

                # ===== 欧式距离分类 =====
                preds_x = torch.argmin(torch.cdist(query_x_trans, proto_x), dim=1)
                preds_f = torch.argmin(torch.cdist(query_fusion, proto_f), dim=1)
                preds_p = torch.argmin(torch.cdist(query_pooled, proto_p), dim=1)

                total_x += (preds_x == query_y).float().mean().item()
                total_f += (preds_f == query_y).float().mean().item()
                total_p += (preds_p == query_y).float().mean().item()

            acc_x_trans.append(total_x)
            acc_fusion.append(total_f)
            acc_pooled.append(total_p)

    mean_x, ci_x, _ = confidence_interval(acc_x_trans)
    mean_f, ci_f, _ = confidence_interval(acc_fusion)
    mean_p, ci_p, _ = confidence_interval(acc_pooled)

    print(f"[x_trans] Accuracy = {mean_x * 100:.2f}% ± {ci_x * 100:.2f}%")
    print(f"[fusion] Accuracy = {mean_f * 100:.2f}% ± {ci_f * 100:.2f}%")
    print(f"[pooled] Accuracy = {mean_p * 100:.2f}% ± {ci_p * 100:.2f}%")

    return (mean_x, ci_x), (mean_f, ci_f), (mean_p, ci_p)

# -----------------------------
# 示例调用
# -----------------------------
if __name__ == "__main__":
    N = 2
    config = dict(
            dataset_name='MIMIIGear',
            N=N, K=5, Q=1, episode_num=1,
            data_path=r"./Dataset/MIMIIGear/",
            mode='', # MultiDomainFeature
            numClass=N,
            numPerClassTrain=10,
            numPerClassTest=200,
            frame_len=4096, # 1024
            num_frames=16,  # 16
        )
    # N = 5
    # config = dict(
    #         dataset_name='pumpJD',
    #         N=N, K=1, Q=1, episode_num=1,
    #         data_path=r"./Dataset/pumpJD/",
    #         mode='MultiDomainFeature',
    #         numClass=N,
    #         numPerClassTrain=10,
    #         numPerClassTest=200,
    #         frame_len=1024,     # 1024
    #         num_frames=16,      # 16
    #     )
    test_loader = data_prepare(config,
                               dataset_name=config.get("dataset_name"),
                               Q_override=config.get('numPerClassTest')-config.get('K'))
    # model_file = "./Model/ModelRegistry/MLFFNet_1shot_JDPump_0.7813_10samples.pth"
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6793_20samples_20251211_000422.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6553_20samples_20251211_001442.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6653_80samples_20251211_012243.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.7553_80samples_20251211_015815.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.7860_200samples_20251211_024922.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6980_200samples_20251211_032108.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.7560_600samples_20251211_040521.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.7820_600samples_20251211_042919.pth",
                   num_repeats=200,
                   test_loader=test_loader)
    # evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.7587_80samples_20251209_150704.pth",
    #                num_repeats=200,
    #                test_loader=test_loader)
    # evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6120_20samples_20251209_153819.pth",
    #                num_repeats=200,
    #                test_loader=test_loader)
    # evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6273_80samples_20251209_162528.pth",
    #                num_repeats=200,
    #                test_loader=test_loader)
    # evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6320_20samples_20251209_171143.pth",
    #                num_repeats=200,
    #                test_loader=test_loader)
    # evaluate_model(model_file="./Model/ModelRegistry/MLFFNet_5shot_MIMIIGear_0.6773_80samples_20251209_171616.pth",
    #                num_repeats=200,
    #                test_loader=test_loader)
