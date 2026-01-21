import numpy as np
from scipy.fftpack import dct
import librosa.display
import scipy.stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import correlate
from scipy.signal import hilbert
from scipy.fft import fftfreq, fft
from scipy import fftpack
class Fea_Extra():
    def __init__(self, Signal, Fs):
        self.signal = Signal
        self.Fs = Fs
    # -------------------------时域特征fea[0:11]------------------------------------
    def Time_fea(self, signal_):
        """
        提取时域特征 11 类
        """
        N = len(signal_)
        y = signal_
        t_mean_1 = np.mean(y)                                    # 1_均值（平均幅值）

        t_std_2  = np.std(y, ddof=1)                             # 2_标准差

        t_fgf_3  = ((np.mean(np.sqrt(np.abs(y)))))**2           # 3_方根幅值

        t_rms_4  = np.sqrt((np.mean(y**2)))                      # 4_RMS均方根

        t_pp_5   = 0.5*(np.max(y)-np.min(y))                     # 5_峰峰值  (参考周宏锑师姐 博士毕业论文)

        #t_skew_6   = np.sum((t_mean_1)**3)/((N-1)*(t_std_3)**3)
        t_skew_6   = scipy.stats.skew(y)                         # 6_偏度 skewness

        #t_kur_7   = np.sum((y-t_mean_1)**4)/((N-1)*(t_std_3)**4)
        t_kur_7 = scipy.stats.kurtosis(y)                        # 7_峭度 Kurtosis

        t_cres_8  = np.max(np.abs(y))/t_rms_4                    # 8_峰值因子 Crest Factor

        t_clear_9  = np.max(np.abs(y))/t_fgf_3                   # 9_裕度因子  Clearance Factor

        t_shape_10 = (N * t_rms_4)/(np.sum(np.abs(y)))           # 10_波形因子 Shape fator

        t_imp_11  = ( np.max(np.abs(y)))/(np.mean(np.abs(y)))  # 11_脉冲指数 Impulse Fator

        t_fea = np.array([t_mean_1, t_std_2, t_fgf_3, t_rms_4, t_pp_5,
                          t_skew_6,   t_kur_7,  t_cres_8,  t_clear_9, t_shape_10, t_imp_11 ])

        #print("t_fea:",t_fea.shape,'\n', t_fea)
        return t_fea

    # ---------------------------------频域特征fea[11:21]--------------------------------------

    def mean_frequency(self):     #求平均频率
        freqs = np.fft.fftfreq(len(self.signal), 1 / self.Fs)
        fft_vals = np.abs(np.fft.fft(self.signal))
        # Compute weighted average of frequencies
        mean_freq = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        return mean_freq

    def frequency_variance(self):     #求频率方差
        freqs = np.fft.fftfreq(len(self.signal), 1 / self.Fs)
        fft_vals = np.abs(np.fft.fft(self.signal))

        # Compute weighted average of frequencies
        mean_freq = np.sum(freqs * fft_vals) / np.sum(fft_vals)

        # Compute variance of frequencies
        var_freq = np.sum((freqs - mean_freq) ** 2 * fft_vals) / np.sum(fft_vals)

        return var_freq

    def peak_frequency(self):     #峰值频率
        freqs, psd = scipy.signal.periodogram(self.signal, self.Fs)
        peaks, _ = find_peaks(psd)
        return freqs[peaks][np.argmax(psd[peaks])]


    def spectral_density(self):   #能量谱密度
        _, psd = welch(self.signal, fs=self.Fs)
        return psd

    def spectral_entropy(self):   #频率熵
        freqs, psd = scipy.signal.periodogram(self.signal, self.Fs)
        norm_psd = psd / np.sum(psd)
        return entropy(norm_psd)

    def rms_power(self):  #均方根功率
        return np.sqrt(np.mean(np.square(self.signal)))

    def autocorr_peak_pos(self):      #自相关函数峰值位置
        corr = correlate(self.signal, self.signal)
        mid = len(corr) // 2
        peak_pos = np.argmax(corr[mid:]) + mid
        return peak_pos

    def mean_power(self):     #平均功率
        fft = np.fft.fft(self.signal)
        return (np.abs(fft) ** 2).mean()

    def power_spectrum_density(self):    #能量频密度
        fft = np.fft.fft(self.signal)
        freqs = np.fft.fftfreq(len(self.signal), 1 / self.Fs)
        return (np.abs(fft) ** 2)[:len(self.signal) // 2] / self.Fs

    def spectral_slope(self):       #频谱斜率
        f, psd = welch(self.signal, fs=self.Fs)
        lpsd = 10 * np.log10(psd)

        # 计算频谱斜率
        slope = np.diff(lpsd) / np.diff(np.log10(f))
        freqs = f[:-1]
        return slope

    def spectral_skewness(self):     #频谱偏度
        fft = np.fft.fft(self.signal)
        freqs = np.fft.fftfreq(len(self.signal), 1 / self.Fs)
        magnitudes = np.abs(fft)[:len(self.signal) // 2]
        mean_freq = np.sum(freqs[:len(magnitudes)] * magnitudes) / np.sum(magnitudes)
        std_freq = np.sqrt(np.sum((freqs[:len(magnitudes)] - mean_freq) ** 2 * magnitudes) / np.sum(magnitudes))

        skewness_num = np.sum(((freqs[:len(magnitudes)] - mean_freq) / std_freq) ** 3 * magnitudes)

        return skewness_num / np.sum(magnitudes)

    def spectral_kurtosis(self):      #谱峭度
        fft = np.fft.fft(self.signal)
        magnitudes = np.abs(fft)[:len(self.signal) // 2]
        mean_mag = magnitudes.mean()
        std_mag = magnitudes.std()

        if std_mag == 0:
            return 0
        normalized_mags = (magnitudes - mean_mag) / std_mag
        kurtosis = (normalized_mags ** 4).mean() - 3
        return kurtosis

    def spectral_envelope(self):      #谱包络
        analytic_signal = hilbert(self.signal)
        envelope = np.abs(analytic_signal)
        return envelope

    def spectral_centroid(self):   #谱质心
        n = len(self.signal)
        freqs = fftfreq(n, 1 / self.Fs)
        fft_signal = fft(self.signal)
        power_spectrum = np.abs(fft_signal) ** 2
        return np.sum(freqs * power_spectrum) / np.sum(power_spectrum)

    def spectral_bandwidth(self):        #谱带宽
        fft = np.abs(fftpack.fft(self.signal))
        freqs = fftpack.fftfreq(len(self.signal)) * self.Fs
        centroid_freq = self.spectral_centroid()

        return np.sqrt(np.sum((freqs - centroid_freq) ** 2 * fft) / np.sum(fft))

    def spectral_energy(self):     #能量谱频率和
        f, psd = welch(self.signal, fs=self.Fs)
        return np.sum(psd)

    def Fre_fea(self, signal_):
        """
        提取频域特征 12类
        :param signal_:
        :return:
        """
        L = len(signal_)
        PL = abs(np.fft.fft(signal_ / L))[: int(L / 2)]
        PL[0] = 0
        f = np.fft.fftfreq(L, 1 / self.Fs)[: int(L / 2)]
        x = f
        y = PL
        K = len(y)

        # f_12 = self.mean_frequency()        #平均频率
        f_12 = np.mean(y)

        # f_13 = self.frequency_variance()       #频率方差
        f_13 = np.var(y)

        # f_14 = self.peak_frequency()    #峰值频率
        f_14 = (np.sum(x * y)) / (np.sum(y))        #重心频率

        f_15 = self.mean_power()                #平均功率
        #!
        f_16 = self.spectral_centroid()   #谱质心
        #!
        f_17 = self.spectral_bandwidth()        #谱带宽

        f_18 = self.spectral_skewness()     #频率偏度

        f_19 = self.spectral_entropy()      #频率熵

        f_20 = self.spectral_kurtosis()             #谱峭度
        #!
        f_21 = self.spectral_energy()             #能量谱频率和

        # f_14 = (np.sum((y - f_12)**3))/(K * ((np.sqrt(f_13))**3))

        # f_15 = (np.sum((y - f_12)**4))/(K * ((f_13)**2))

        # f_16 = (np.sum(x * y))/(np.sum(y))    #重心频率

        # f_17 = np.sqrt((np.mean(((x- f_16)**2)*(y))))     #频率加权标准差

        # f_18 = np.sqrt((np.sum((x**2)*y))/(np.sum(y)))    #功率加权平均值

        # f_19 = np.sqrt((np.sum((x**4)*y))/(np.sum((x**2)*y)))

        # f_20 = (np.sum((x**2)*y))/(np.sqrt((np.sum(y))*(np.sum((x**4)*y))))

        # f_21 = f_17/f_16

        f_22 = (np.sum(((x - f_16)**3)*y))/(K * (f_17**3))  #谱偏度

        f_23 = (np.sum(((x - f_16)**4)*y))/(K * (f_17**4))  #谱峰度




        #print("f_16:",f_16)

        # f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24])
        f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21,f_22,f_23])

        #print("f_fea:",f_fea.shape,'\n', f_fea)
        return f_fea

    # --------------------------------------------MFCC特征------------------------------------------
    def MFCC(self,signal_):
        # 归一化倒谱提升窗口
        lifts = []
        for n in range(1, 13):
            lift = 1 + 6 * np.sin(np.pi * n / 12)
            lifts.append(lift)
        # print(lifts)
        yf = np.abs(np.fft.fft(signal_))
        # print(yf.shape)
        # 谱线能量
        yf = yf ** 2
        # 梅尔滤波器系数
        nfilt = 24
        low_freq_mel = 0
        NFFT = 256
        high_freq_mel = (2595 * np.log10(1 + (self.Fs / 2) / 700))  # 把 Hz 变成 Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 将梅尔刻度等间隔
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 把 Mel 变成 Hz
        bin = np.floor((NFFT + 1) * hz_points / self.Fs)
        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(yf[0:129], fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
        filter_banks = 10 * np.log10(filter_banks)  # dB
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        # print(filter_banks)
        # DCT系数
        num_ceps = 12
        c2 = dct(filter_banks, type=2, axis=-1, norm='ortho')[1: (num_ceps + 1)]  # Keep 2-13
        c2 *= lifts
        return c2


        

    #-----------------------------------------------频域绘图------------------------------------------
    def frequency(self,Fs):
        N = 512
        Y = fft(self, N)
        p = np.abs(Y)  # 双侧频谱
        p = p / max(p)
        signal = 20 * np.log10(p)
        f = np.arange(0, N / 2 - 1) * Fs / N
        plt.plot(f, signal[:255])
        plt.title('Single-Sided Amplitude Spectrum of X(t)')
        plt.xlabel('f (Hz)')
        plt.ylabel('|P1(f)|')
        plt.show()



    # --------------------提取时频特性-----------------------------------
    def Both_Fea(self):
        """
        :return: 时域、频域特征 array
        """
        t_fea = self.Time_fea(self.signal)
        f_fea = self.Fre_fea(self.signal)
        m_fea = self.MFCC(self.signal)

        fea = np.append(np.array(t_fea), np.array(f_fea))
        fea = np.append(fea,np.array(m_fea))
        #print("fea:", fea.shape, '\n', fea)
        return fea

