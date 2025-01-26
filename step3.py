import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt,lfilter
import tkinter as tk
from tkinter import filedialog

# 创建一个隐藏的 Tkinter 根窗口
root = tk.Tk()
root.withdraw()  # 隐藏根窗口

##从步骤二得到的结果中导入系数
fir_coefficients = np.loadtxt('FIRnum.txt')
iir_num=np.loadtxt("IIRnum.txt")
iir_den=np.loadtxt('IIRden.txt')
base_num=np.loadtxt('basenum.txt')
base_den=np.loadtxt('baseden.txt')
##导入步骤一中得到的数据
# 弹窗选择输入文件路径，默认文件路径可以设置为 'channel_1_data.txt', 'channel_2_data.txt', 'channel_3_data.txt'
file_path1 = filedialog.askopenfilename(title="选择 Channel 1 数据文件", initialdir='.', filetypes=[("Text files", "*.txt")]) or 'channel_1_data.txt'
file_path2 = filedialog.askopenfilename(title="选择 Channel 2 数据文件", initialdir='.', filetypes=[("Text files", "*.txt")]) or 'channel_2_data.txt'
file_path3 = filedialog.askopenfilename(title="选择 Channel 3 数据文件", initialdir='.', filetypes=[("Text files", "*.txt")]) or 'channel_3_data.txt'

channel1_data = np.loadtxt(file_path1)
channel2_data = np.loadtxt(file_path2)
channel3_data = np.loadtxt(file_path3)

t1 = np.linspace(0, 1, len(channel1_data))  # 使用 channel1_data 的长度
t2 = np.linspace(0, 1, len(channel2_data))  # 使用 channel2_data 的长度
t3 = np.linspace(0, 1, len(channel3_data))
##自定义FIR滤波函数
def fir_filter(b, x):
    # 正向滤波
    y_forward = np.convolve(b, x)

    # 反向滤波
    y_backward = np.convolve(b, y_forward[::-1])[::-1]

    # 计算延迟补偿索引
    delay = (len(b) - 1) // 2

    # 调整长度以匹配输入数据
    start_idx = delay
    end_idx = start_idx + len(x)
    filtered_data = y_backward[start_idx:end_idx]

    return filtered_data

##自定义IIR滤波函数
def iir_filter(input_signal, b, a):
    """
    自定义的IIR滤波器函数
    :param input_signal: 输入信号
    :param b: 前向滤波器系数
    :param a: 后向滤波器系数
    :return: 滤波后的信号
    """
    # 初始化输出信号数组，长度与输入信号相同
    output_signal = np.zeros_like(input_signal)
    # 前向滤波
    x = np.zeros(len(b))
    for n in range(len(input_signal)):
        # 更新前向滤波器状态
        x = np.roll(x, 1)
        x[0] = input_signal[n]
        y = np.dot(b, x)
        for i in range(1, len(a)):
            y -= a[i] * output_signal[n - i]
        output_signal[n] = y
    # 后向滤波
    x = np.zeros(len(b))
    output_signal_rev = output_signal[::-1]
    output_signal = np.zeros_like(input_signal)
    for n in range(len(output_signal_rev)):
        # 更新后向滤波器状态
        x = np.roll(x, 1)
        x[0] = output_signal_rev[n]
        y = np.dot(b, x)
        for i in range(1, len(a)):
            y -= a[i] * output_signal[n - i]
        output_signal[n] = y
    return output_signal[::-1]

##高通滤波器去除直流分量（基线漂移）
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

##使用自定义的（IIR滤波函数）进行滤波channel1
filtered_data1 = iir_filter(channel1_data,iir_num,iir_den)
# 使用自定义的filtfilt（FIR滤波函数）进行零相位滤波--channel2
filtered_data2 = fir_filter(fir_coefficients, channel2_data)
##使用自定义的（IIR滤波函数）去工频干扰50Hz
filtered_data3_1 = iir_filter(channel3_data,iir_num,iir_den)

# channel1截取前 99% 的数据
percent_99_index = int(len(channel1_data) * 0.993)
channel1_data = channel1_data[:percent_99_index]
filtered_data1 = filtered_data1[:percent_99_index]
t1 = t1[:percent_99_index]

# channel2_data去除前50个数据
filtered_data2 = filtered_data2[50:]

# channel3(未去基线漂移）截取前 99% 的数据
percent_99_index = int(len(channel3_data) * 0.993)
channel3_data = channel3_data[:percent_99_index]
filtered_data3_1 = filtered_data3_1[:percent_99_index]
t3 = t3[:percent_99_index]

fs = 200  # 采样频率
cutoff = 1  # 截止频率
b, a = butter_highpass(cutoff, fs)
##使用高通滤波器去直流分量（基线漂移）
filtered_data3 = filtfilt(base_num,base_den,filtered_data3_1)

# channel3(去基线漂移）截取前 99% 的数据
percent_99_index = int(len(filtered_data3_1) * 0.993)
filtered_data3 = filtered_data3[:percent_99_index]
t4 = t3[:percent_99_index]

channel3_data=channel3_data[10:-50]
filtered_data3_1 = filtered_data3_1[10:-50]
t3 = t3[10:-50]
filtered_data3=filtered_data3[50:-70]
t4=t4[50:-70]

##channel1通道绘图
plt.subplot(7, 1, 1)
plt.plot(t1, channel1_data, label='原始数据')
plt.title('channel1_original')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(7, 1, 2)
plt.plot(t1, filtered_data1, label='滤波后数据', color='orange')  # 调整时间轴
plt.title('channel1_filtered')
plt.xlabel('time')
plt.ylabel('amplitude')

##channel2通道绘图
plt.subplot(7, 1, 3)
plt.plot(t2, channel2_data, label='原始数据')
plt.title('channel2_original')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(7, 1, 4)
plt.plot(t2[50:], filtered_data2, label='滤波后数据', color='orange')  # 调整时间轴
plt.title('channel2_filtered')
plt.xlabel('time')
plt.ylabel('amplitude')

##channel1通道绘图
plt.subplot(7, 1, 5)
plt.plot(t3, channel3_data, label='原始数据')
plt.title('channel3_original')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(7, 1, 6)
plt.plot(t3, filtered_data3_1, label='滤波后数据', color='orange')  # 调整时间轴
plt.title('channel3_filtered_50Hz')
plt.xlabel('time')
plt.ylabel('amplitude')

##channel3通道（去直流）绘图
plt.subplot(7, 1, 7)
plt.plot(t4, filtered_data3, label='原始数据')
plt.title('channel3_filtered_base')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.show()

# 选择保存文件的路径
save_path1 = filedialog.asksaveasfilename(title="保存 Channel 1 数据", defaultextension=".txt", filetypes=[("Text files", "*.txt")]) or 'filtered_channel1_data.txt'
save_path2 = filedialog.asksaveasfilename(title="保存 Channel 2 数据", defaultextension=".txt", filetypes=[("Text files", "*.txt")]) or 'filtered_channel2_data.txt'
save_path3 = filedialog.asksaveasfilename(title="保存 Channel 3 数据", defaultextension=".txt", filetypes=[("Text files", "*.txt")]) or 'filtered_channel3_data.txt'

# 保存处理后的数据
np.savetxt(save_path1, filtered_data1)
np.savetxt(save_path2, filtered_data2)
np.savetxt(save_path3, filtered_data3)