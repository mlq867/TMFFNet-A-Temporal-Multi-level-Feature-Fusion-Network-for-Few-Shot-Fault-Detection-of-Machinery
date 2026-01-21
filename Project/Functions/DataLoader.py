import random

import numpy as np
import torch

from FeatureExtraction.multiDomainFeatureExtraction.Feature import Fea_Extra

class Dataset:
    """
    数据集类，用于管理训练集和测试集
    """
    def __init__(self, path, datasetName, numPerClassTrain=30, numPerClassTest=500,
                 mode=None, frame_len=1024, num_frames=16):
        self.path = path
        self.datasetName = datasetName
        self.numPerClassTrain = numPerClassTrain
        self.numPerClassTest = numPerClassTest
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.mode = mode
        if path is not None:
            self.X_train = np.load(path + 'X_train.npy')
            self.y_train = np.load(path + 'y_train.npy')
            self.X_test = np.load(path + 'X_test.npy')
            self.y_test = np.load(path + 'y_test.npy')
            self.X_train, self.y_train = self.balanceByLabel(self.X_train, self.y_train, self.numPerClassTrain)
            self.X_test, self.y_test = self.balanceByLabel(self.X_test, self.y_test, self.numPerClassTest)
        if mode == "MultiDomainFeature":
            self.X_train = self.mutiDomainFeaturesMaker(self.X_train, frame_len=frame_len, num_frames=num_frames)
            self.X_test = self.mutiDomainFeaturesMaker(self.X_test, frame_len=frame_len, num_frames=num_frames)
        mean = np.mean(self.X_train, axis=0)
        std = np.std(self.X_train, axis=0)
        self.X_train = (self.X_train - mean) / std
        self.X_test = (self.X_test - mean) / std

    def balanceByLabel(self, X, y, maxPerClass=None):
        """让每个标签的数据量相同，取最小类别样本数或指定数量"""
        X_balanced, y_balanced = [], []
        labels = np.unique(y)
        counts = [np.sum(y == lbl) for lbl in labels]
        minCount = min(counts) if maxPerClass is None else min(max(counts), maxPerClass)
        for lbl in labels:
            indices = np.where(y == lbl)[0]
            if len(indices) > minCount:
                indices = np.random.choice(indices, minCount, replace=False)
            X_balanced.append(X[indices])
            y_balanced.append(y[indices])

        X_balanced = np.concatenate(X_balanced, axis=0)
        y_balanced = np.concatenate(y_balanced, axis=0)
        return X_balanced, y_balanced

    def mutiDomainFeaturesExtraction(self, frame):
        # 归一化和预处理在构建数据集时已经做了
        feature_voice = Fea_Extra(frame, 48000)
        fea = feature_voice.Both_Fea()
        return fea

    def mutiDomainFeaturesMaker(self, data, frame_len=1024, num_frames=8):
        """
        将 (N, 2048) 数据转换为 (N, num_frames, 35)

        Args:
            data: numpy array or torch tensor, shape [N, 2048]
            frame_len: 每帧长度
            num_frames: 输出帧数（固定）

        Returns:
            features: torch tensor, shape [N, num_frames, 35]
        """
        N, total_len = data.shape
        step = (total_len - frame_len) // (num_frames - 1)  # 自动计算步长保证 num_frames
        if step > frame_len:
            raise ValueError('step > frame_len')
        all_features = []

        for i in range(N):
            row = data[i]
            frames = []
            for j in range(num_frames):
                start = j * step
                end = start + frame_len
                frame = row[start:end]
                if len(frame) < frame_len:  # 不够长度则零填充
                    frame = np.pad(frame, (0, frame_len - len(frame)))
                feat = self.mutiDomainFeaturesExtraction(frame)  # [35]
                frames.append(feat)
            frames = np.stack(frames, axis=0)  # [num_frames, 35]
            all_features.append(frames)

        all_features = np.stack(all_features, axis=0)  # [N, num_frames, 35]
        return all_features
