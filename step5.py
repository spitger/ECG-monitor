import sys
import serial
import threading
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import serial.tools.list_ports
from PyQt6.QtWidgets import QFileDialog  # 导入QFileDialog
from PyQt6.QtWidgets import QMessageBox
import pandas as pd
def FIR_filter50hz(data):
    fir_coefficients = np.array([
        -0.000000, 0.000031, 0.000194, -0.000157, -0.000926, 0.000000,
        0.002242, 0.000991, -0.003872, -0.003358, 0.005184, 0.007474,
        -0.005209, -0.013407, 0.002688, 0.020835, 0.003944, -0.029060,
        -0.016903, 0.037105, 0.040719, -0.043884, -0.091224, 0.048412,
        0.313168, 0.450024, 0.313168, 0.048412, -0.091224, -0.043884,
        0.040719, 0.037105, -0.016903, -0.029060, 0.003944, 0.020835,
        0.002688, -0.013407, -0.005209, 0.007474, 0.005184, -0.003358,
        -0.003872, 0.000991, 0.002242, 0.000000, -0.000926, -0.000157,
        0.000194, 0.000031, -0.000000,
    ])
    return np.convolve(data, fir_coefficients, mode='same')

def FIR_filter60hz(data):
    fir_coefficients = np.array([
        -0.000000, 0.000031, 0.000194, -0.000157, -0.000926, 0.000000,
        0.002242, 0.000991, -0.003872, -0.003358, 0.005184, 0.007474,
        -0.005209, -0.013407, 0.002688, 0.020835, 0.003944, -0.029060,
        -0.016903, 0.037105, 0.040719, -0.043884, -0.091224, 0.048412,
        0.313168, 0.450024, 0.313168, 0.048412, -0.091224, -0.043884,
        0.040719, 0.037105, -0.016903, -0.029060, 0.003944, 0.020835,
        0.002688, -0.013407, -0.005209, 0.007474, 0.005184, -0.003358,
        -0.003872, 0.000991, 0.002242, 0.000000, -0.000926, -0.000157,
        0.000194, 0.000031, -0.000000,
    ])
    return np.convolve(data, fir_coefficients, mode='same')

##计算心率相关代码移植
def calculate_derivative(ecg_signal):
    derivative = np.abs(np.diff(ecg_signal, 2))  # 计算二阶差分
    return derivative

def calculate_threshold(derivative, sampling_rate):
    two_seconds_samples = int(2 * sampling_rate)
    buffer = derivative[:two_seconds_samples]
    P = np.max(buffer)
    threshold = 0.7 * P
    return threshold

def detect_qrs_peaks(derivative, threshold, max_search_window=40, skip_window=50):
    qrs_peaks = []
    i = 0
    while i < len(derivative) - max_search_window:
        if derivative[i] > threshold:
            search_window = derivative[i:i + max_search_window]
            M1_index = np.argmax(search_window)
            M1_value = search_window[M1_index]
            qrs_peaks.append(i + M1_index)
            i += skip_window
        else:
            i += 1
    return qrs_peaks

def calculate_heart_rate(rr_intervals, sampling_rate):
    heart_rates = []
    for i in range(0, len(rr_intervals) - 4):
        window_rr_intervals = rr_intervals[i:i + 5]
        avg_rr_interval = np.mean(window_rr_intervals)
        heart_rate = (60 * sampling_rate) / avg_rr_interval
        heart_rates.append(heart_rate)
    return heart_rates


class Oscilloscope(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Step 1: 获取最大数据点数
        self.max_points, ok = QtWidgets.QInputDialog.getInt(self, '输入数据点数',
                                                            '请输入最大数据点数 (建议大于100):',
                                                            1000, 100, 10000, 1)

        if not ok:
            self.max_points = 1000  # 如果用户取消了，设置默认值

        # Step 2: 获取 COM 口
        self.serial_port_name = self.select_com_port()

        self.setWindowTitle("Real-Time Oscilloscope (Progressive Clear Mode)")
        self.resize(800, 600)

        # Parameters
        self.current_index = 0
        self.clear_width = 10
        self.tag1 = 0
        self.tag2 = 0

        # Data buffers
        self.channel_data1 = np.zeros(self.max_points)
        self.channel_data2 = np.zeros(self.max_points)
        self.channel_data3 = np.zeros(self.max_points)
        self.sampling_rate = 200  # 假设采样率为200 Hz

        # PyQtGraph setup
        self.plot_widget1 = pg.PlotWidget(title="Channel 1")
        self.plot_widget2 = pg.PlotWidget(title="Channel 2")
        self.plot_widget3 = pg.PlotWidget(title="Channel 3")
        self.plot_widget1.setBackground('w')
        self.plot_widget2.setBackground('w')
        self.plot_widget3.setBackground('w')

        # Create one plot for each channel
        self.curve1 = self.plot_widget1.plot(pen='b', name="Channel 1")
        self.heart_rate_label1 = QtWidgets.QLabel("心率：0 次/分钟")
        self.curve2 = self.plot_widget2.plot(pen='b', name="Channel 2")
        self.heart_rate_label2 = QtWidgets.QLabel("心率：0 次/分钟")
        self.curve3 = self.plot_widget3.plot(pen='b', name="Channel 3")
        self.heart_rate_label3 = QtWidgets.QLabel("心率：0 次/分钟")

        # Axis ranges
        self.plot_widget1.setXRange(0, self.max_points - 1, padding=0)
        self.plot_widget1.setYRange(0, 1023, padding=0)

        self.plot_widget2.setXRange(0, self.max_points - 1, padding=0)
        self.plot_widget2.setYRange(0, 1023, padding=0)

        self.plot_widget3.setXRange(0, self.max_points - 1, padding=0)
        self.plot_widget3.setYRange(100, 600, padding=0)

        # Buttons for toggling filters
        self.filter_button1 = QtWidgets.QPushButton('滤除50赫兹')
        self.filter_button2 = QtWidgets.QPushButton('滤除60赫兹')
        self.filter_button3 = QtWidgets.QPushButton('去心电干扰')

        # Button to save data
        self.save_button = QtWidgets.QPushButton('保存数据')

        # New buttons for Start receiving, Pause receiving and Clear data
        self.start_button = QtWidgets.QPushButton('开始接收')
        self.pause_button = QtWidgets.QPushButton('暂停接收')
        self.clear_button = QtWidgets.QPushButton('清空数据')
        self.filter_button = QtWidgets.QPushButton('一键滤波')
        self.calculate_heart_rate_button = QtWidgets.QPushButton("计算心率")

        # Connecting buttons to filter toggling functions
        self.filter_button1.clicked.connect(self.toggle_filter1)
        self.filter_button2.clicked.connect(self.toggle_filter2)
        self.filter_button3.clicked.connect(self.toggle_filter3)
        self.save_button.clicked.connect(self.save_data)  # 连接保存按钮
        self.start_button.clicked.connect(self.start_receiving)  # 开始接收数据按钮
        self.pause_button.clicked.connect(self.toggle_receiving)  # 暂停接收按钮
        self.clear_button.clicked.connect(self.clear_data)  # 清空数据按钮
        self.filter_button.clicked.connect(self.toggle_filter1)
        self.filter_button.clicked.connect(self.toggle_filter2)
        self.filter_button.clicked.connect(self.toggle_filter3)
        self.calculate_heart_rate_button.clicked.connect(self.calculate_heart_rate_tag)

        # Layout setup: Stack three plots and their respective buttons horizontally
        layout = QtWidgets.QVBoxLayout()

        # Create horizontal layouts for each channel + button
        group1_layout = QtWidgets.QVBoxLayout()
        group1_layout.addWidget(self.plot_widget1, 1)
        group1_layout.addWidget(self.heart_rate_label1)

        layout1 = QtWidgets.QHBoxLayout()
        layout1.addLayout(group1_layout)
        layout1.addWidget(self.filter_button1)

        group2_layout = QtWidgets.QVBoxLayout()
        group2_layout.addWidget(self.plot_widget2, 1)
        group2_layout.addWidget(self.heart_rate_label2)

        layout2 = QtWidgets.QHBoxLayout()
        layout2.addLayout(group2_layout)
        layout2.addWidget(self.filter_button2)

        group3_layout = QtWidgets.QVBoxLayout()
        group3_layout.addWidget(self.plot_widget3, 1)
        group3_layout.addWidget(self.heart_rate_label3)

        layout3 = QtWidgets.QHBoxLayout()
        layout3.addLayout(group3_layout)
        layout3.addWidget(self.filter_button3)

        # Add the horizontal layouts to the main vertical layout
        layout.addLayout(layout1)
        layout.addLayout(layout2)
        layout.addLayout(layout3)

        # Create a horizontal layout for the start, save, and clear buttons
        layout_buttons = QtWidgets.QVBoxLayout()
        layout_buttons.addWidget(self.start_button)
        layout_buttons.addWidget(self.pause_button)
        layout_buttons.addWidget(self.filter_button)
        layout_buttons.addWidget(self.calculate_heart_rate_button)
        layout_buttons.addWidget(self.save_button)
        layout_buttons.addWidget(self.clear_button)

        layout.addLayout(layout_buttons)

        self.setLayout(layout)

        # Serial port setup (start with closed port)
        self.serial_port = None
        self.serial_thread = None

        # Timer for updating the plot
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # Update every 50 ms

        # Filter states
        self.filter_enabled1 = False
        self.filter_enabled2 = False
        self.filter_enabled3 = False

        # Variable to track whether data reception is started or paused
        self.receiving_data = False
        self.paused = False  # New variable to track the pause state

    def select_com_port(self):
        # Get the list of available COM ports
        ports = serial.tools.list_ports.comports()
        port_names = [port.device for port in ports]

        if not port_names:
            # 如果没有可用的串口
            QMessageBox.warning(self, "Warning", "未检测到可用的串口设备。", QMessageBox.StandardButton.Ok)
            return None

        # 显示COM口选择对话框
        com_port, ok = QtWidgets.QInputDialog.getItem(self, '选择COM口', '请选择串口:', port_names, 0, False)

        if ok:
            return com_port
        else:
            return None

    def start_receiving(self):
        if self.serial_port and self.receiving_data and not self.paused:
            QMessageBox.warning(self, "Warning", "Already receiving data.", QMessageBox.StandardButton.Ok)
            return

        # Open serial port and start receiving data
        if self.serial_port_name:
            try:
                self.serial_port = serial.Serial(self.serial_port_name, 115200, timeout=1)
                self.receiving_data = True
                self.paused = False
                self.serial_thread = threading.Thread(target=self.read_serial_data)
                self.serial_thread.daemon = True
                self.serial_thread.start()
            except serial.SerialException as e:
                QMessageBox.critical(self, "Error", f"Failed to open serial port: {e}", QMessageBox.StandardButton.Ok)

    def update_data(self):
        if self.receiving_data:
            # 实时更新数据（模拟接收过程）
            # 在实际应用中，这里应该是通过串口或其他方式接收数据
            new_data1 = np.random.randn()  # 示例数据生成
            new_data2 = np.random.randn()
            new_data3 = np.random.randn()

            # 更新通道数据
            self.channel_data1 = np.roll(self.channel_data1, -1)
            self.channel_data2 = np.roll(self.channel_data2, -1)
            self.channel_data3 = np.roll(self.channel_data3, -1)
            self.channel_data1[-1] = new_data1
            self.channel_data2[-1] = new_data2
            self.channel_data3[-1] = new_data3

            # 更新绘图
            self.curve1.setData(self.channel_data1)
            self.curve2.setData(self.channel_data2)
            self.curve3.setData(self.channel_data3)
    def calculate_heart_rate_tag(self):
        if self.tag1:
            self.tag1=0
        else:
            self.tag1=1
    def calculate_heart_rate(self):
        # 计算每个通道的心率
        for channel_data, heart_rate_label in [(self.channel_data1, self.heart_rate_label1),
                                               (self.channel_data2, self.heart_rate_label2),
                                               (self.channel_data3, self.heart_rate_label3)]:
            # 1. 计算信号的一阶导数
            derivative = calculate_derivative(channel_data)
            # 2. 计算阈值
            threshold = calculate_threshold(derivative, self.sampling_rate)
            # 3. 检测QRS波峰
            qrs_peaks = detect_qrs_peaks(derivative, threshold)
            # 4. 计算RR间期
            rr_intervals = np.diff(qrs_peaks)
            # 5. 计算心率
            heart_rates = calculate_heart_rate(rr_intervals, self.sampling_rate)

            if heart_rates:
                avg_heart_rate = np.mean(heart_rates)
                heart_rate_label.setText(f"心率：{avg_heart_rate:.0f} 次/分钟")
    def toggle_receiving(self):
        if self.paused:
            self.paused = False  # Resume receiving data
            self.pause_button.setText('暂停接收')  # Change button text back to "Pause"
        else:
            self.paused = True  # Pause receiving data
            self.pause_button.setText('恢复接收')  # Change button text to "Resume"

    def clear_data(self):
        # Clear data buffers
        self.channel_data1.fill(0)
        self.channel_data2.fill(0)
        self.channel_data3.fill(0)
        self.current_index = 0
        QMessageBox.information(self, "Success", "Data cleared successfully.", QMessageBox.StandardButton.Ok)

    def toggle_filter1(self):
        self.filter_enabled1 = not self.filter_enabled1

    def toggle_filter2(self):
        self.filter_enabled2 = not self.filter_enabled2

    def toggle_filter3(self):
        self.filter_enabled3 = not self.filter_enabled3

    def read_serial_data(self):
        while self.receiving_data:
            if self.paused:  # If paused, wait before trying to read again
                continue

            try:
                line = self.serial_port.readline().decode('ascii').strip()
                if line:
                    values = list(map(int, line.split(',')))
                    if len(values) == 3:
                        # Update the data buffers for each channel
                        self.channel_data1[self.current_index] = values[0]
                        self.channel_data2[self.current_index] = values[1]
                        self.channel_data3[self.current_index] = values[2]

                        # Clear data segment
                        clear_start = (self.current_index + 1) % self.max_points
                        clear_end = (self.current_index + 1 + self.clear_width) % self.max_points
                        if clear_start < clear_end:
                            self.channel_data1[clear_start:clear_end] = 0
                            self.channel_data2[clear_start:clear_end] = 0
                            self.channel_data3[clear_start:clear_end] = 0
                        else:
                            self.channel_data1[clear_start:] = 0
                            self.channel_data2[clear_start:] = 0
                            self.channel_data3[clear_start:] = 0
                            self.channel_data1[:clear_end] = 0
                            self.channel_data2[:clear_end] = 0
                            self.channel_data3[:clear_end] = 0

                        # Update the current index
                        self.current_index += 1
                        if self.current_index >= self.max_points:
                            self.current_index = 0
            except (ValueError, serial.SerialException):
                pass

    def update_plot(self):
        # Apply filter if enabled
        if self.filter_enabled1:
            data1 = FIR_filter50hz(self.channel_data1)
        else:
            data1 = self.channel_data1

        if self.filter_enabled2:
            data2 = FIR_filter60hz(self.channel_data2)
        else:
            data2 = self.channel_data2

        if self.filter_enabled3:
            data3 = FIR_filter50hz(self.channel_data3)
        else:
            data3 = self.channel_data3

        # Create the X data for the sweep
        x_data = np.arange(self.max_points)

        # Update the plot for each channel
        self.curve1.setData(x_data, data1)
        self.curve2.setData(x_data, data2)
        self.curve3.setData(x_data, data3)

        if self.tag1:
            self.calculate_heart_rate()

    def save_data(self):
        # Open file dialog to choose the save path and file format
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Data", "", "Text Files (*.txt);;CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if file_path:
            # If user selected a path, save the data
            try:
                # Stack the three channels' data together
                data_to_save = np.vstack((self.channel_data1, self.channel_data2, self.channel_data3)).T

                # Save the data in the selected format
                if selected_filter == "Text Files (*.txt)":
                    # Save as TXT file
                    np.savetxt(file_path, data_to_save, delimiter='\t', header="Channel 1\tChannel 2\tChannel 3",
                               comments='')

                elif selected_filter == "CSV Files (*.csv)":
                    # Save as CSV file using pandas for better CSV handling
                    df = pd.DataFrame(data_to_save, columns=["Channel 1", "Channel 2", "Channel 3"])
                    df.to_csv(file_path, index=False)

                elif selected_filter == "Excel Files (*.xlsx)":
                    # Save as Excel file using pandas
                    df = pd.DataFrame(data_to_save, columns=["Channel 1", "Channel 2", "Channel 3"])
                    df.to_excel(file_path, index=False)

                QMessageBox.information(self, "Success", "Data saved successfully!", QMessageBox.StandardButton.Ok)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {e}", QMessageBox.StandardButton.Ok)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    oscilloscope = Oscilloscope()
    oscilloscope.show()
    sys.exit(app.exec())