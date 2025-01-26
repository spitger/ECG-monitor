import numpy as np
import matplotlib.pyplot as plt


# 读取txt文件中的心电图数据，假设数据是以空格或换行分隔的一维数组
def load_ecg_data(file_path):
    return np.loadtxt(file_path)


# 计算信号的一阶导数
def calculate_derivative(ecg_signal):
    # 使用公式 y0(n) = |x(n + 1) - x(n - 1)|
    derivative = np.abs(np.diff(ecg_signal, 2))  # 计算二阶差分
    return derivative


# 阈值计算
def calculate_threshold(derivative, sampling_rate):
    # 存储前两秒的数据并获取最大值
    two_seconds_samples = int(2 * sampling_rate)
    buffer = derivative[:two_seconds_samples]
    P = np.max(buffer)
    threshold = 0.7 * P
    return threshold


# 检测QRS波峰
def detect_qrs_peaks(derivative, threshold, max_search_window=40, skip_window=50):
    qrs_peaks = []
    i = 0
    while i < len(derivative) - max_search_window:
        if derivative[i] > threshold:
            # 搜索最大值
            search_window = derivative[i:i + max_search_window]
            M1_index = np.argmax(search_window)
            M1_value = search_window[M1_index]
            qrs_peaks.append(i + M1_index)
            i += skip_window  # 跳过50个样本，避免重复检测
        else:
            i += 1
    return qrs_peaks


# 计算心率，去除第一个心率波形
def calculate_heart_rate(rr_intervals, sampling_rate):
    heart_rates = []
    # 从第二个RR间期开始计算（即索引1），避免第一个误差较大的心率
    for i in range(1, len(rr_intervals) - 4):  # 滑动窗口，计算每五个RR间期的平均心率
        window_rr_intervals = rr_intervals[i:i + 5]
        avg_rr_interval = np.mean(window_rr_intervals)  # 计算当前窗口的平均RR间期
        # 使用公式：HR = (60 * 采样率) / 平均RR间期
        heart_rate = (60 * sampling_rate) / avg_rr_interval  # 计算心率
        heart_rates.append(heart_rate)
    return heart_rates


def main(file_path, sampling_rate=200):
    # 1. 读取心电图数据
    ecg_signal = load_ecg_data(file_path)

    # 2. 计算一阶导数
    derivative = calculate_derivative(ecg_signal)

    # 3. 计算阈值
    threshold = calculate_threshold(derivative, sampling_rate)

    # 4. 检测QRS波峰
    qrs_peaks = detect_qrs_peaks(derivative, threshold)

    # 5. 计算RR间期
    rr_intervals = np.diff(qrs_peaks)  # 计算RR间期，即连续QRS波峰之间的间隔

    # 6. 计算多个心率
    heart_rates = calculate_heart_rate(rr_intervals, sampling_rate)

    # 7. 输出多个心率（修改为一行输出）
    print(f"多个心率计算结果（次/分钟）：")
    heart_rate_str = ', '.join([f"{hr:.2f}" for hr in heart_rates])  # 格式化心率为两位小数，并用逗号连接
    print(heart_rate_str)  # 一行输出所有心率

    # 8. 计算并输出平均心率
    average_heart_rate = np.mean(heart_rates)
    print(f"平均心率：{average_heart_rate:.2f} 次/分钟")

    plt.figure(figsize=(64, 18))  # 调整图形的大小，宽度12英寸，高度18英寸
    # 可选：绘制信号和检测到的QRS波峰
    plt.plot(ecg_signal)

    plt.title("ECG Signal with Detected QRS Peaks (Excluding First Peak)")
    plt.show()

file_path1 = 'filtered_channel1_data.txt'
main(file_path1)

file_path2 = 'filtered_channel2_data.txt'
main(file_path2)

file_path3 = 'filtered_channel3_data.txt'  # 替换为你的文件路径
main(file_path3)
