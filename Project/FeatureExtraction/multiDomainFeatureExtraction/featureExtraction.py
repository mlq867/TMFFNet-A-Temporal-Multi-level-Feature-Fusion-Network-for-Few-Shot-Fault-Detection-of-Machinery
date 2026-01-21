import os
import librosa
import PreProcess
from Feature import Fea_Extra
import scipy.io.wavfile as wav
import wave
import Feature
import pandas as pd
import struct
import binascii
import numpy as np


def wav_mean_handle(path):
    dat, fs = librosa.load(path, sr=None, mono=False)
    x = np.array(dat.T, dtype=np.float32)
    # print(x.shape)
    x = np.mean(x, axis=1)
    # print(x.shape)
    return x, fs


path_list = [r'./sound/rawData/Normal/',
             r'./sound/rawData/Abnormal/',]

save_list = [r'./sound/feature/Normal/',
             r'./sound/feature/Abnormal/']

if __name__ == '__main__':
    for c in range(len(path_list)):
        path = path_list[c]
        for num, file in enumerate(os.listdir(path), 1):
            file_data, file_rate = wav_mean_handle(path + file)
            # file_data_one = file_data[0, :]
            file_data_one = file_data
            file_data_one = file_data_one * 1.0 / (max(abs(file_data_one)))
            file_data_one = PreProcess.pre_fun(file_data_one)
            # print(file_data_one.shape)
            frame_data, _, _ = PreProcess.frame(file_data_one, 4096, 2048)
            temp1 = np.empty((0, 35))  # 0-11时域，11-23频域，23-35MFCC
            for i in range(len(frame_data)):
                feature_voice = Feature.Fea_Extra(frame_data[i, :], file_rate)
                fea = feature_voice.Both_Fea()
                temp1 = np.vstack((temp1, fea))
            temp_path = save_list[c] + file.split('.wav')[0] + '.npy'
            np.save(temp_path, temp1)
            print(f'{file} saved, size : {temp1.shape}')
            # np.savetxt(temp_path, temp1, delimiter=',')
