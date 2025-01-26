import serial
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from tqdm import tqdm

# 全局变量
PORT = None
BAUDRATE = 115200  # 默认的波特率
TIMEOUT = 1
SAMPLE_COUNT = None
CHANNEL_1_FILE = None
CHANNEL_2_FILE = None
CHANNEL_3_FILE = None

def choose_port_and_settings():
    global PORT, BAUDRATE, SAMPLE_COUNT
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 选择串口
    PORT = simpledialog.askstring("选择串口", "请输入串口号 (如 COM5):")
    if not PORT:
        messagebox.showerror("错误", "没有选择串口，程序退出。")
        exit()

    # 选择比特率
    baud_rate = simpledialog.askinteger("选择比特率", "请输入串口的波特率:", initialvalue=BAUDRATE, minvalue=9600, maxvalue=921600)
    if baud_rate:
        BAUDRATE = baud_rate
    else:
        messagebox.showerror("错误", "没有选择波特率，程序退出。")
        exit()

    # 选择数据长度
    SAMPLE_COUNT = simpledialog.askinteger("接收数据", "请输入接收数据的点数:", minvalue=1, maxvalue=10000)
    if not SAMPLE_COUNT:
        messagebox.showerror("错误", "没有选择数据长度，程序退出。")
        exit()

    # 关闭对话框
    root.quit()

def choose_save_path_and_format():
    global CHANNEL_1_FILE, CHANNEL_2_FILE, CHANNEL_3_FILE
    # 文件保存路径选择
    save_path = filedialog.askdirectory(title="选择保存路径")
    if not save_path:
        messagebox.showerror("错误", "没有选择保存路径，程序退出。")
        exit()

    # 文件保存格式
    file_format = simpledialog.askstring("保存格式", "请输入文件保存格式（如 .txt 或 .csv）:")
    if not file_format:
        messagebox.showerror("错误", "没有选择保存格式，程序退出。")
        exit()

    # 设置文件路径
    CHANNEL_1_FILE = f"{save_path}/channel_1_data{file_format}"
    CHANNEL_2_FILE = f"{save_path}/channel_2_data{file_format}"
    CHANNEL_3_FILE = f"{save_path}/channel_3_data{file_format}"

def receive_and_store_data():
    # 初始化串口
    ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)

    # 计时
    start_time = time.time()

    # 数据存储列表
    channel_1 = []
    channel_2 = []
    channel_3 = []

    # 进度条设置
    pbar = tqdm(total=SAMPLE_COUNT, desc="接收数据", ncols=100)

    try:
        while len(channel_1) < SAMPLE_COUNT:
            # 从串口读取一行数据
            line = ser.readline().decode('ascii').strip()
            if line:  # 确保数据不为空
                values = line.split(',')  # 分割数据
                if len(values) == 3:  # 确保有3个通道数据
                    try:
                        ch1, ch2, ch3 = int(values[0]), int(values[1]), int(values[2])
                        channel_1.append(ch1)
                        channel_2.append(ch2)
                        channel_3.append(ch3)
                        pbar.update(1)  # 更新进度条
                    except ValueError:
                        print(f"无效数据被忽略: {line}")
    except KeyboardInterrupt:
        print("用户中断接收。")
    finally:
        # 关闭串口
        ser.close()
        pbar.close()
        print("数据接收完成，正在保存数据...")

    end_time = time.time()
    execution_time = end_time - start_time
    print("数据接收完成！")
    return channel_1, channel_2, channel_3, execution_time

def plot_data(channel_1, channel_2, channel_3):
    # 绘制曲线图
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(channel_1, label='Channel 1')
    plt.title('Channel 1 Data')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')

    plt.subplot(3, 1, 2)
    plt.plot(channel_2, label='Channel 2', color='orange')
    plt.title('Channel 2 Data')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')

    plt.subplot(3, 1, 3)
    plt.plot(channel_3, label='Channel 3', color='green')
    plt.title('Channel 3 Data')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 选择串口和设置
    choose_port_and_settings()

    # 接收数据
    channel_1_data, channel_2_data, channel_3_data, execution_time = receive_and_store_data()

    # 选择保存路径和格式（移动到数据接收后）
    choose_save_path_and_format()

    # 将数据保存到文件
    with open(CHANNEL_1_FILE, 'w') as f1, open(CHANNEL_2_FILE, 'w') as f2, open(CHANNEL_3_FILE, 'w') as f3:
        f1.write("\n".join(map(str, channel_1_data)))
        f2.write("\n".join(map(str, channel_2_data)))
        f3.write("\n".join(map(str, channel_3_data)))

    # 绘制曲线
    plot_data(channel_1_data, channel_2_data, channel_3_data)

    # 输出程序执行时间
    print(f"程序执行时间: {execution_time:.2f} 秒")
