%%IIR50赫兹低通滤波器
clear; clc;
% 设定参数
fs = 200;             % 采样频率 200 Hz
fc = 40;              % 截止频率 40 Hz
order = 50;           % 滤波器阶数

% 计算归一化截止频率
Wc = fc / (fs / 2);  

% 设计低通滤波器
[b, a] = butter(order, Wc, 'low');

% 保存b、a向量到文本文件
writematrix(b, 'IIRnum.txt', 'Delimiter', '\t'); % 保存b向量
writematrix(a, 'IIRden.txt', 'Delimiter', '\t'); % 保存a向量

% 绘制滤波器的幅频特性
[H, F] = freqz(b, a, 1024, fs);  % 计算频率响应，1024点
subplot(611)
plot(F, abs(H));
title('IIR50赫兹低通滤波器幅频特性');
xlabel('频率 (Hz)');
ylabel('幅值');
grid on;

subplot(612)
plot(F, unwrap(angle(H)));
title('IIR50赫兹低通滤波器相频特性');
xlabel('频率 (Hz)');
ylabel('幅值');
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%FIR60赫兹带阻滤波器
clear; clc;

f_s = 200;   % 采样频率 200 Hz
f_0 = 60;    % 带阻中心频率 60 Hz
N = 50;      % 滤波器阶次

% 归一化频率
f_low = (f_0 - 5) / (f_s / 2);   % 低频边界 55 Hz
f_high = (f_0 + 5) / (f_s / 2);  % 高频边界 65 Hz

% 使用fir1设计FIR带阻滤波器
b = fir1(N, [f_low, f_high], 'stop');

% 保存滤波器系数到文件
writematrix(b, 'FIRnum.txt', 'Delimiter', '\t');

% 计算频率响应
[H, f] = freqz(b, 1, 1024, f_s);  % freqz返回滤波器频率响应，1024点，Fs为采样频率

% 绘制幅频特性
subplot(613)
plot(f, abs(H));  % 绘制幅度响应
xlabel('频率 (Hz)');
ylabel('幅值');
title('FIR60赫兹带阻滤波器幅频特性');
grid on;

subplot(614)
plot(f, unwrap(angle(H)));  % 绘制幅度响应
xlabel('频率 (Hz)');
ylabel('幅值');
title('FIR60赫兹带阻滤波器相频特性');
grid on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%设置高通滤波器
clear;clc

fs = 200; % 采样频率 (Hz)
cutoff = 1; % 截止频率 (Hz)

% 调用滤波器设计函数
[b, a] = butter_highpass(cutoff, fs);

% 保存b、a向量到文本文件
writematrix(b, 'basenum.txt', 'Delimiter', '\t'); % 保存b向量
writematrix(a, 'baseden.txt', 'Delimiter', '\t'); % 保存a向量

% 绘制滤波器的幅频特性
[H, F] = freqz(b, a, 1024, fs);  % 计算频率响应，1024点
subplot(615);
plot(F, abs(H));
title('IIR高通滤波器幅频特性');
xlabel('频率 (Hz)');
ylabel('幅值');
grid on;

subplot(616);
plot(F, unwrap(angle(H)));
title('IIR高通滤波器相频特性');
xlabel('频率 (Hz)');
ylabel('幅值');
grid on;
function [b, a] = butter_highpass(cutoff, fs, order)
    if nargin < 3
        order = 5;  % 默认阶数为5
    end
    
    nyquist = 0.5 * fs;  % 奈奎斯特频率
    normal_cutoff = cutoff / nyquist;  % 归一化截止频率
    
    % 设计高通滤波器
    [b, a] = butter(order, normal_cutoff, 'high');
end