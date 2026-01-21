import os
import random

import librosa
import scipy.io.wavfile as wav
import wave
import pandas as pd
import numpy as np
import pandas as pd

from FeatureExtraction.multiDomainFeatureExtraction.PreProcess import pre_fun, frame
from FeatureExtraction.multiDomainFeatureExtraction.Feature import Fea_Extra

def wavRead(path):
    dat, fs = librosa.load(path, sr=None, mono=False)
    x = np.array(dat.T, dtype=np.float32)
    # print(x.shape)
    # x = np.mean(x, axis=1)
    # print(x.shape)
    return x, fs


def findAllWavFiles(rootDir, trainRatio=0.3, seed=42):
    wavFiles = []
    for dirPath, _, fileNames in os.walk(rootDir):
        for fileName in fileNames:
            if fileName.lower().endswith('.wav'):
                wavFiles.append(os.path.join(dirPath, fileName))
    # 打乱顺序
    random.seed(seed)
    random.shuffle(wavFiles)

    # 按比例划分
    split_idx = int(len(wavFiles) * trainRatio)
    trainFiles = wavFiles[:split_idx]
    testFiles = wavFiles[split_idx:]

    print(f"总文件数: {len(wavFiles)} | 训练集: {len(trainFiles)} | 测试集: {len(testFiles)}")
    return trainFiles, testFiles


def frameSignal(signal, frame_len=2048, hop_len=1024):
    """将信号分帧"""
    frames = []
    for start in range(0, len(signal) - frame_len + 1, hop_len):
        frames.append(signal[start:start + frame_len])
    return np.array(frames)


def balanceByLabel(X, y, maxPerClass=None):
    """让每个标签的数据量相同，取最小类别样本数或指定数量"""
    X_balanced, y_balanced = [], []
    labels = np.unique(y)
    counts = [np.sum(y == lbl) for lbl in labels]

    if maxPerClass is None:
        minCount = min(counts)
    elif maxPerClass == -1:
        minCount = max(counts)
    else:
        minCount = min(max(counts), maxPerClass)

    for lbl in labels:
        indices = np.where(y == lbl)[0]
        if len(indices) > minCount:
            indices = np.random.choice(indices, minCount, replace=False)
        X_balanced.append(X[indices])
        y_balanced.append(y[indices])

    X_balanced = np.concatenate(X_balanced, axis=0)
    y_balanced = np.concatenate(y_balanced, axis=0)
    return X_balanced, y_balanced


def pumpJDHandle():
    pathList = [r'E:\声纹数据集\江大数据（泵）\pumpRAB\normal\\',
                r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_25\\',
                r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_50\\',
                r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_75\\',
                r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_25_75\\']
    saveDir = r'../Dataset/pumpJD/'
    X_train, X_test = [], []
    y_train, y_test = [], []
    for c in range(len(pathList)):
        path = pathList[c]
        print(f'正在处理{path}...')
        trainFiles, testFiles = findAllWavFiles(path, trainRatio=0.3, seed=42)
        for trainFile in trainFiles:
            fileData, sampleRate = wavRead(trainFile)
            fileData = fileData * 1.0 / max(abs(fileData))
            fileData = pre_fun(fileData)
            frames = frameSignal(fileData, frame_len=2048, hop_len=1024)
            X_train.append(frames)
            y_train.append(np.full((frames.shape[0],), c))
        for testFile in testFiles:
            fileData, sampleRate = wavRead(testFile)
            fileData = fileData * 1.0 / max(abs(fileData))
            fileData = pre_fun(fileData)
            frames = frameSignal(fileData, frame_len=2048, hop_len=2048)
            X_test.append(frames)
            y_test.append(np.full((frames.shape[0],), c))

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    random.seed(30)
    print(f'X_train = {X_train.shape}, X_test = {X_test.shape}')
    X_train, y_train = balanceByLabel(X_train, y_train, 2000)
    X_test, y_test = balanceByLabel(X_test, y_test, 2000)
    balancedCounts = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"各类别平衡后训练样本数: {balancedCounts}")
    balancedCounts = dict(zip(*np.unique(y_test, return_counts=True)))
    print(f"各类别平衡后测试样本数: {balancedCounts}")

    np.save(os.path.join(saveDir, 'X_train.npy'), X_train)
    np.save(os.path.join(saveDir, 'y_train.npy'), y_train)
    np.save(os.path.join(saveDir, 'X_test.npy'), X_test)
    np.save(os.path.join(saveDir, 'y_test.npy'), y_test)
    print(f"✅ 数据保存完成到 {saveDir}")


def pumpJDHandleAll():
    import numpy as np
    import os, glob, random

    pathList = [
        r'E:\声纹数据集\江大数据（泵）\pumpRAB\normal\\',
        r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_25\\',
        r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_50\\',
        r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_75\\',
        r'E:\声纹数据集\江大数据（泵）\pumpRAB\abnormal_25_75\\'
    ]

    saveDir = r'../Dataset/pumpJD/ALL_'
    # os.makedirs(saveDir, exist_ok=True)

    train_ratio = 0.3   # 按帧划分比例

    X_train, y_train = [], []
    X_test,  y_test = [], []

    for c, clsPath in enumerate(pathList):
        print(f"正在处理类别 {c}: {clsPath}")

        wavFiles = glob.glob(os.path.join(clsPath, "*.wav"))
        if len(wavFiles) == 0:
            print("⚠ 警告：未找到wav文件")
            continue

        # 保存该类别所有帧
        all_frames = []

        for f in wavFiles:
            data, sr = wavRead(f)
            data = data / max(abs(data))
            data = pre_fun(data)

            # 使用训练帧参数（2048/1024）
            frames = frameSignal(data, 2048, 1024)  # shape (num_frames, 2048)
            all_frames.append(frames)

        # 合并为一个大数组
        all_frames = np.concatenate(all_frames, axis=0)
        num_frames = all_frames.shape[0]
        print(f"类别 {c} 共帧数：{num_frames}")

        # 随机打乱帧索引
        idx = np.arange(num_frames)
        np.random.shuffle(idx)

        split = int(num_frames * train_ratio)
        train_idx = idx[:split]
        test_idx  = idx[split:]

        # 按帧划分
        X_train.append(all_frames[train_idx])
        y_train.append(np.full((len(train_idx),), c))

        X_test.append(all_frames[test_idx])
        y_test.append(np.full((len(test_idx),), c))

    # 合并所有类别的数据
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test  = np.concatenate(X_test, axis=0)
    y_test  = np.concatenate(y_test, axis=0)

    # 打乱最终数据（可选）
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    perm = np.random.permutation(len(X_test))
    X_test, y_test = X_test[perm], y_test[perm]

    # 保存
    np.save(os.path.join(saveDir, "X_train.npy"), X_train)
    np.save(os.path.join(saveDir, "y_train.npy"), y_train)
    np.save(os.path.join(saveDir, "X_test.npy"),  X_test)
    np.save(os.path.join(saveDir, "y_test.npy"),  y_test)

    print("保存完成")
    print("训练集分布:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("测试集分布:", dict(zip(*np.unique(y_test, return_counts=True))))

import pandas as pd

def read_xlsx_column(file_path, column_name, skip_first_row=False):
    """
    读取 Excel 文件中指定列的数据。

    参数：
        file_path (str): Excel 文件路径。
        column_name (str 或 int): 列名（如 'A列' 对应的标题）或列索引（从 0 开始）。
        skip_first_row (bool): 是否去除第一行内容（通常为表头或无效数据）。

    返回：
        list: 指定列的数据（已去除 NaN）。
    """
    df = pd.read_excel(file_path)

    # 根据列名或索引取列
    if isinstance(column_name, int):
        data = df.iloc[:, column_name]
    else:
        data = df[column_name]

    # 去除第一行
    if skip_first_row:
        data = data.iloc[1:]

    data = pd.to_numeric(data, errors='coerce')
    # 去除缺失值并转为列表
    return data.dropna().tolist()


def CRBHandle():
    pathList = [r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\Defect free-table2\Defect free\Acoustic',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\INNER RACE-table2\INNER RACE\IR-I\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\INNER RACE-table2\INNER RACE\IR-II\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\INNER RACE-table2\INNER RACE\IR-III\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\INNER RACE-table2\INNER RACE\IR-IV\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\OUTER RACE-table2\OUTER RACE\OR-I\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\OUTER RACE-table2\OUTER RACE\OR-II\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\OUTER RACE-table2\OUTER RACE\OR-III\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\OUTER RACE-table2\OUTER RACE\OR-IV\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\ROLLER-table2\ROLLER\RO-I\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\ROLLER-table2\ROLLER\RO-II\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\ROLLER-table2\ROLLER\RO-III\ACCOUSTIC\data',
                r'E:\声纹数据集\圆柱滚子轴承缺陷情况的振动和声学数据\ROLLER-table2\ROLLER\RO-IV\ACCOUSTIC\data',]
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for i, folder in enumerate(pathList):
        print(f"读取第 {i} 类文件夹: {folder}")
        for file in os.listdir(folder):
            if file.endswith('.xlsx'):
                file_path = os.path.join(folder, file)
                try:
                    # 读取第一列并去除第一行
                    col_data = read_xlsx_column(file_path, 0, skip_first_row=True)
                    col_data = np.array(col_data, dtype=float)
                    fileData = col_data * 1.0 / max(abs(col_data))
                    fileData = pre_fun(fileData)
                    frames = frameSignal(fileData, frame_len=2048, hop_len=1024)

                    random.seed(42)
                    random.shuffle(frames)

                    # 按比例划分
                    split_idx = int(len(frames) * 0.5)
                    trainData = frames[:split_idx]
                    testData = frames[split_idx:]

                    train_data.append(trainData)
                    test_data.append(testData)
                    train_label.append([i] * len(trainData))
                    test_label.append([i] * len(testData))
                    print(f"  已读取文件: {file}, 训练样本数: {len(trainData)}， 测试样本数：{len(testData)}")
                except Exception as e:
                    print(f"  读取失败: {file}，错误: {e}")

    X_train = np.concatenate(train_data, axis=0)
    X_test = np.concatenate(test_data, axis=0)
    y_train = np.concatenate(train_label, axis=0)
    y_test = np.concatenate(test_label, axis=0)
    print(f"总训练样本数：{len(train_data)}  {len(train_label)}, 总测试样本数：{len(test_data)}  {len(test_label)}")

    random.seed(30)
    X_train, y_train = balanceByLabel(X_train, y_train, 2000)
    X_test, y_test = balanceByLabel(X_test, y_test, 2000)
    balancedCounts = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"各类别平衡后训练样本数: {balancedCounts}")
    balancedCounts = dict(zip(*np.unique(y_test, return_counts=True)))
    print(f"各类别平衡后测试样本数: {balancedCounts}")
    saveDir = r'../Dataset/CRB/'
    np.save(os.path.join(saveDir, 'X_train.npy'), X_train)
    np.save(os.path.join(saveDir, 'y_train.npy'), y_train)
    np.save(os.path.join(saveDir, 'X_test.npy'), X_test)
    np.save(os.path.join(saveDir, 'y_test.npy'), y_test)
    print(f"✅ 数据保存完成到 {saveDir}")


def MIMIIGearHandle(folder_path, save_dir, N_per_class_train=2000, N_per_class_test=1000, train_ratio=0.5):
    random.seed(42)

    label_map = {
        "normal": 0,
        "anomaly": 1
    }

    class_files = {0: [], 1: []}

    # ----------------------------------------------------
    # 1. 遍历文件夹
    # ----------------------------------------------------
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(folder_path, fname)
        fname_low = fname.lower()
        matched = False
        for key, label in label_map.items():
            if key in fname_low:
                class_files[label].append(fpath)
                matched = True
                break
        if not matched:
            print(f"⚠️ 跳过无法识别标签的文件：{fname}")

    # ----------------------------------------------------
    # 2. 分帧 + 特征提取（每帧35维）
    # ----------------------------------------------------
    frame_len = 2048
    hop = 1024
    T = 50      # 每个样本16帧
    step_T = 5  # 0% 覆盖

    all_samples = []
    all_labels = []

    print("开始处理所有文件...")
    for label, files in class_files.items():
        for fpath in files:
            fileData, fs = wavRead(fpath)

            # 归一化
            m = max(abs(fileData))
            if m > 0:
                fileData = fileData / m

            # 预加重
            fileData = pre_fun(fileData)

            # 按 2048/1024 分帧
            frames = frameSignal(fileData, frame_len=frame_len, hop_len=hop)
            n_frames = frames.shape[0]

            # 每帧提取 35 维特征
            fea_list = []
            for i in range(n_frames):
                feat = Fea_Extra(frames[i], fs).Both_Fea()  # [35]
                fea_list.append(feat)

            fea_arr = np.stack(fea_list, axis=0)  # [n_frames, 35]

            # ----------------------------------------------------
            # 3. 连续16帧构成1个样本（步长=8）
            # ----------------------------------------------------
            for start in range(0, n_frames - T + 1, step_T):
                sample = fea_arr[start:start + T]  # [16, 35]
                all_samples.append(sample)
                all_labels.append(label)

    # 转成数组
    all_samples = np.array(all_samples)       # [N, 16, 35]
    all_labels = np.array(all_labels)         # [N]

    print("总样本数：", all_samples.shape[0])
    print("标签分布：", dict(zip(*np.unique(all_labels, return_counts=True))))

    # ----------------------------------------------------
    # 4. 打乱 + 划分训练/测试
    # ----------------------------------------------------
    idx = np.arange(len(all_labels))
    np.random.seed(42)
    np.random.shuffle(idx)

    split = int(len(idx) * train_ratio)
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = all_samples[train_idx]
    y_train = all_labels[train_idx]
    X_test = all_samples[test_idx]
    y_test = all_labels[test_idx]

    print("训练集样本：", X_train.shape[0])
    print("测试集样本：", X_test.shape[0])

    # ----------------------------------------------------
    # 5. 样本平衡（保持你原来的逻辑）
    # ----------------------------------------------------
    X_train, y_train = balanceByLabel(X_train, y_train, N_per_class_train)
    X_test, y_test = balanceByLabel(X_test, y_test, N_per_class_test)

    print("平衡后训练样本：", dict(zip(*np.unique(y_train, return_counts=True))))
    print("平衡后测试样本：", dict(zip(*np.unique(y_test, return_counts=True))))

    # ----------------------------------------------------
    # 6. 保存结果
    # ----------------------------------------------------
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

    print(f"✅ 数据保存完成到 {save_dir}")

def MIMIIGearHandle_2(
    folder_path,
    save_dir,
    N_per_class_train=2000,
    N_per_class_test=1000,
    train_ratio=0.5
):
    random.seed(42)
    np.random.seed(42)

    label_map = {
        "normal": 0,
        "anomaly": 1
    }

    class_files = {0: [], 1: []}

    # ----------------------------------------------------
    # 1. 遍历文件夹，按文件名确定标签
    # ----------------------------------------------------
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".wav"):
            continue

        fpath = os.path.join(folder_path, fname)
        fname_low = fname.lower()
        matched = False

        for key, label in label_map.items():
            if key in fname_low:
                class_files[label].append(fpath)
                matched = True
                break

        if not matched:
            print(f"⚠️ 跳过无法识别标签的文件：{fname}")

    # ----------------------------------------------------
    # 2. 原始信号切成定长序列
    # ----------------------------------------------------
    seq_len = 2048 * 16      # 一个样本长度（点数）
    step = seq_len // 2     # 50% 重叠

    all_samples = []
    all_labels = []

    print("开始处理所有文件...")
    for label, files in class_files.items():
        for fpath in files:
            fileData, fs = wavRead(fpath)

            # 归一化
            m = np.max(np.abs(fileData))
            if m > 0:
                fileData = fileData / m

            # 预加重（如果你想用纯 raw，可直接注释）
            fileData = pre_fun(fileData)

            total_len = len(fileData)

            # 滑窗切片
            for start in range(0, total_len - seq_len + 1, step):
                sample = fileData[start:start + seq_len]   # [seq_len]
                all_samples.append(sample)
                all_labels.append(label)

    # ----------------------------------------------------
    # 3. 转 numpy
    # ----------------------------------------------------
    all_samples = np.array(all_samples)   # [N, seq_len]
    all_labels = np.array(all_labels)     # [N]

    print("总样本数：", all_samples.shape[0])
    print("标签分布：", dict(zip(*np.unique(all_labels, return_counts=True))))

    # 如果你后面模型需要通道维（CNN / Transformer）
    # all_samples = all_samples[..., np.newaxis]  # [N, seq_len, 1]

    # ----------------------------------------------------
    # 4. 打乱 + 划分训练 / 测试
    # ----------------------------------------------------
    idx = np.arange(len(all_labels))
    np.random.shuffle(idx)

    split = int(len(idx) * train_ratio)
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = all_samples[train_idx]
    y_train = all_labels[train_idx]
    X_test = all_samples[test_idx]
    y_test = all_labels[test_idx]

    print("训练集样本：", X_train.shape[0])
    print("测试集样本：", X_test.shape[0])

    # ----------------------------------------------------
    # 5. 样本平衡（保持你原来的逻辑）
    # ----------------------------------------------------
    X_train, y_train = balanceByLabel(X_train, y_train, N_per_class_train)
    X_test, y_test = balanceByLabel(X_test, y_test, N_per_class_test)

    print("平衡后训练样本：", dict(zip(*np.unique(y_train, return_counts=True))))
    print("平衡后测试样本：", dict(zip(*np.unique(y_test, return_counts=True))))

    # ----------------------------------------------------
    # 6. 保存结果
    # ----------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

    print(f"✅ 数据保存完成到 {save_dir}")


if __name__ == '__main__':
    # pumpJDHandle()
    # CRBHandle()
    # MIMIIGearHandle(r'E:\声纹数据集\MIMII DUE\dev_data_gearbox\gearbox\target_test\\',
    #                 r'../Dataset/MIMIIGear/',
    #                 4000,
    #                 1000,
    #                 0.8)
    MIMIIGearHandle_2(r'E:\声纹数据集\MIMII DUE\dev_data_gearbox\gearbox\target_test\\',
                    r'../Dataset/MIMIIGearRaw/',
                    4000,
                    1000,
                    0.8)
