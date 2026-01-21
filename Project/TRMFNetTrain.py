import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from types import SimpleNamespace
from Functions.DataLoader import Dataset
from Functions.normTranDataLoader import FewShotDataset
from Net.TRMFNet import Transformer_ResNetMultiLevelFeatureFusionNet
from ModelIndexSystem import ModelIndex  # 动态 ModelIndex

index_file = r'C:\Users\86198\Desktop\少样本\Code\fewShotProject\Model\TRMFNet-2\model_index.json'

# -----------------------------
# 数据加载
# -----------------------------
def get_data_loaders(cfg):
    dataset = Dataset(
        path=cfg.data_path,
        datasetName=cfg.dataset_name,
        mode=cfg.mode,
        numPerClassTrain=cfg.numPerClassTrain,
        numPerClassTest=cfg.numPerClassTest
    )

    X_train = torch.tensor(dataset.X_train, dtype=torch.float32)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long)
    X_test = torch.tensor(dataset.X_test, dtype=torch.float32)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long)

    train_dataset = FewShotDataset(X_train, y_train, episode_num=cfg.episode_num,
                                   N=cfg.N, K=cfg.K, Q=cfg.Q)
    test_dataset = FewShotDataset(X_test, y_test, episode_num=cfg.episode_num,
                                  N=cfg.N, K=cfg.K, Q=cfg.Q)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

# -----------------------------
# 测试函数
# -----------------------------
def test_fewshot(model, test_loader, device):
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(5):
            total_acc = 0
            for support_x, support_y, query_x, query_y in test_loader:
                support_x = support_x.squeeze(0).transpose(1, 2).to(device)
                support_y = support_y.squeeze(0).to(device)
                query_x = query_x.squeeze(0).transpose(1, 2).to(device)
                query_y = query_y.squeeze(0).to(device)

                logits, _ = model(query_x, support_x, support_y)
                preds = torch.argmax(logits, dim=1)
                total_acc += (preds == query_y).float().mean().item()
            accs.append(total_acc / len(test_loader))
    accs = np.array(accs)
    return float(np.mean(accs))

# -----------------------------
# 单轮训练
# -----------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for support_x, support_y, query_x, query_y in train_loader:
        support_x = support_x.squeeze(0).transpose(1,2).to(device)
        support_y = support_y.squeeze(0).to(device)
        query_x = query_x.squeeze(0).transpose(1,2).to(device)
        query_y = query_y.squeeze(0).to(device)

        optimizer.zero_grad()
        logits, _ = model(query_x, support_x, support_y)
        loss = criterion(logits, query_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

# -----------------------------
# 主函数
# -----------------------------
def main(cfg):
    device = cfg.device
    train_loader, test_loader = get_data_loaders(cfg)

    model = Transformer_ResNetMultiLevelFeatureFusionNet(cfg.model_params,
                                                         cfg.model,
                                                         cfg.isTopDown,
                                                         cfg.isMultiLevel).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model_index = ModelIndex(index_file)  # 初始化模型索引管理器
    best_acc = 0.0
    best_model_id = -1  # 用于 ModelIndex 的更新
    model_file = ''
    for epoch in range(cfg.num_epochs):
        start_time = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        elapsed = time.time() - start_time
        acc = test_fewshot(model, test_loader, device)
        print(f"=>Epoch {epoch+1}/{cfg.num_epochs} | Loss={loss:.4f} | Test Acc={acc:.4f} | Time={elapsed:.1f}s")

        scheduler.step()

        if acc > best_acc:
            if epoch >= 1:
                os.remove(model_file)
            best_acc = acc
            os.makedirs(cfg.weight_path, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            n = cfg.model + f"_{cfg.K}shot_{cfg.dataset_name}_{best_acc:.4f}_{cfg.numPerClassTrain}samples_{timestamp}.pth"
            model_file = os.path.join(cfg.weight_path, n)
            torch.save(model.state_dict(), model_file)
            print(f"=> Saved new best model: {model_file}")

            # 展开完整 config
            full_config = vars(cfg).copy()
            full_config.update(full_config.pop('model_params', {}))

            # 更新 ModelIndex
            best_model_id = model_index.addModel(
                model_id=best_model_id,  # -1 表示新增，已有 id 表示更新
                file_path=model_file,
                accuracy=best_acc,
                timestamp=timestamp,
                **full_config
            )


if __name__ == "__main__":
    config = SimpleNamespace(
        # 数据集相关
        dataset_name='MIMIIGear',  # 数据集名称
        data_path='./Dataset/MIMIIGear/',  # 数据集位置
        mode='',  # 数据处理方法（多域特征）
        frame_len=4096,  # no use
        num_frames=16,  # no use
        # 模型相关
        N=2,  # 类别
        K=1,  # shot（每次每类支持集数据个数）
        Q=1,  # 每类查询集个数
        episode_num=150,  # 每次循环训练次数
        num_epochs=60,  # epochs
        numPerClassTrain=2,  # 每类训练数据量
        numPerClassTest=100,  # 每类测试数据量
        device='cpu',  # 设备
        model_params=dict(  # 参数
            d_model=35,
            d_ff=2048,
            d_k=64,
            d_v=64,
            n_layers=4,
            n_heads=1,
            n_class=2,  # 与类别必须一致
            n_sequence=50,
            distance_metric="euclidean",

            target_channel=64,
            layer1_channel=64,
            layer2_channel=128,
            layer3_channel=256,
            layer4_channel=512,
            layer1_num=1,
            layer2_num=1,
            layer3_num=1,
            layer4_num=1,
            dropout=0.2,
        ),
        lr=1e-4,  # lr
        weight_path='./Model/TRMFNet-2/',  # 模型保存的路径

        model='MLFFNet',  # MLFFNet, Trans, MultiLevel
        isTopDown=True,
        isMultiLevel=5,
        note=''
    )
    # config = SimpleNamespace(
    #     # 数据集相关
    #     dataset_name='JDPump',  # 数据集名称
    #     data_path='./Dataset/pumpJD/',  # 数据集位置
    #     mode='MultiDomainFeature',  # 数据处理方法（多域特征）
    #     frame_len=1024,
    #     num_frames=16,
    #     # 模型相关
    #     N=5,  # 类别
    #     K=1,  # shot（每次每类支持集数据个数）
    #     Q=1,  # 每类查询集个数
    #     episode_num=150,  # 每次循环训练次数
    #     num_epochs=60,  # epochs
    #     numPerClassTrain=2,  # 每类训练数据量
    #     numPerClassTest=100,  # 每类测试数据量
    #     device='cpu',  # 设备
    #     model_params=dict(  # 参数
    #         d_model=35,
    #         d_ff=2048,
    #         d_k=64,
    #         d_v=64,
    #         n_layers=4,
    #         n_heads=1,
    #         n_class=5,  # 与类别必须一致
    #         n_sequence=16,
    #         distance_metric="euclidean",
    #
    #         target_channel=64,
    #         layer1_channel=64,
    #         layer2_channel=128,
    #         layer3_channel=256,
    #         layer4_channel=512,
    #         layer1_num=1,
    #         layer2_num=1,
    #         layer3_num=1,
    #         layer4_num=1,
    #         dropout=0.2,
    #     ),
    #     lr=1e-4,  # lr
    #     weight_path='./Model/TRMFNet-2/',  # 模型保存的路径
    #
    #     model='MLFFNet',  # MLFFNet, Trans, MultiLevel
    #     isTopDown=True,
    #     isMultiLevel=5,
    #     note=''
    # )
    config.K = 5
    config.num_epochs = 20
    # config.model = 'MultiLevel'
    # config.isTopDown = False

    config.numPerClassTrain = 10
    main(config)
    config.numPerClassTrain = 20
    main(config)
    config.numPerClassTrain = 100
    main(config)
    config.numPerClassTrain = 200
    main(config)

