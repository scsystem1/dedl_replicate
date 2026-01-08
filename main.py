import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 导入自定义模块
import config
from utils import FNN_asig, calculate_mse
from data_generation import get_data, generate_x, generate_t, generate_y_true
from dedl import calculate_ate


def train_model(net, train_loader, optimizer, criterion):
    """
    执行一个 Epoch 的训练
    """
    net.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        optimizer.zero_grad()
        outputs = net(inputs)

        # 计算 L1 正则化 (如 notebook 所示)
        l1_norm = sum(p.abs().sum() for p in net.parameters())
        loss = criterion(outputs, labels) + config.reg_loss * l1_norm

        loss.backward()
        optimizer.step()


def plot_figure7(history, save_path="Figure7_Replication.pdf"):
    """
    绘制并保存复现的 Figure 7
    """
    epochs = history['epoch']
    mse = history['train_mse']

    # 转换为百分比
    sdl_mape = np.array(history['mape_sdl']) * 100
    dedl_mape = np.array(history['mape_dedl']) * 100
    lr_mape = np.array(history['mape_lr']) * 100

    # 设置绘图风格 (复刻论文风格)
    sns.set_theme(style="whitegrid",
                  font="Times New Roman" if "Times New Roman" in plt.rcParams['font.family'] else "sans-serif")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴: Training MSE (红色)
    color_mse = 'tab:red'
    ax1.set_xlabel('Training Epoch', fontsize=14)
    ax1.set_ylabel('Training MSE', color=color_mse, fontsize=14)
    line1, = ax1.plot(epochs, mse, color=color_mse, marker='.', markersize=4, linestyle='-', label='Training MSE',
                      alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.set_ylim(bottom=0)
    ax1.grid(False)  # 关闭左轴网格，避免混乱

    # 右轴: Estimation MAPE (蓝色/绿色/虚线等)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Estimation MAPE (%)', color='black', fontsize=14)

    # 绘制三条 MAPE 线
    line2, = ax2.plot(epochs, dedl_mape, color='navy', linestyle='-', linewidth=2, label='DeDL MAPE')
    line3, = ax2.plot(epochs, sdl_mape, color='gray', linestyle='--', linewidth=2, label='SDL MAPE')
    line4, = ax2.plot(epochs, lr_mape, color='gray', linestyle=':', linewidth=2, alpha=0.8, label='LR MAPE')

    ax2.tick_params(axis='y', labelcolor='black')
    # 设置右轴范围，确保能看清 DeDL 的下降趋势 (参考 Figure 7)
    # 通常 MAPE 在 0% 到 100% 之间，或者根据数据自适应
    top_lim = max(np.max(sdl_mape), np.max(dedl_mape), np.max(lr_mape)) * 1.2
    ax2.set_ylim(0, top_lim)

    # 合并图例
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, fontsize=12)

    plt.title('Replication of Figure 7: MAPE Comparison with DNN Training Epoch', fontsize=16, pad=20)
    plt.tight_layout()

    # 保存为 PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()


def main():
    print(f"Starting reproduction of Figure 7...")
    print(f"Device: {config.device}")

    # 1. 准备数据
    # train_loader 用于训练
    # data_info 包含 Ground Truth 参数和组合信息
    train_loader, data_info = get_data()

    # 生成用于评估的 Estimation Set (观测数据)
    # 这一步是为了模拟论文中计算 ATE 时使用的评估数据集 (Samples for Inference)
    np.random.seed(config.seed)  # 确保评估数据固定
    x_est = generate_x(config.d_c, config.n_est)
    t_est = generate_t(data_info['t_combo_obs'], data_info['t_dist_obs'], config.n_est)
    y_est, _ = generate_y_true(
        data_info['coef'], data_info['c_true'], data_info['d_true'],
        x_est, t_est, config.n_est
    )

    # 2. 初始化模型与优化器
    net = FNN_asig().to(config.device)

    # Warm-up heuristic: 初始化 c 参数 (layer3 weight) 为 Outcome 的最大值附近
    # 这有助于 sigmoid 快速收敛到正确的量级
    with torch.no_grad():
        net.layer3.weight[0, 0] = float(np.max(y_est))

    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wd)
    criterion = nn.MSELoss()

    # 3. 记录训练过程
    history = {
        'epoch': [],
        'train_mse': [],
        'mape_sdl': [],
        'mape_dedl': [],
        'mape_lr': []
    }

    # 4. 训练循环
    print(f"Training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        # 训练一步
        train_model(net, train_loader, optimizer, criterion)

        # 计算 Training MSE
        train_mse = calculate_mse(train_loader, net)

        # 计算 MAPE (DeDL, SDL, LR)
        # 这里的计算比较耗时，但为了绘制平滑的曲线，我们在每一轮都计算
        # 传入 x_est, t_est, y_est 作为 "观测到的评估集"
        # 传入 data_info['t_combo'] 作为所有需要预测的 Treatment 组合
        mape_sdl, mape_dedl, mape_lr = calculate_ate(
            net, x_est, t_est, y_est, data_info['t_combo'], data_info
        )

        # 记录数据
        history['epoch'].append(epoch)
        history['train_mse'].append(train_mse)
        history['mape_sdl'].append(mape_sdl)
        history['mape_dedl'].append(mape_dedl)
        history['mape_lr'].append(mape_lr)

        # 打印进度 (每 10 epoch)
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d} | Train MSE: {train_mse:.4f} | "
                  f"MAPE - DeDL: {mape_dedl:.2%}, SDL: {mape_sdl:.2%}, LR: {mape_lr:.2%}")

    # 5. 绘图并保存
    plot_figure7(history)


if __name__ == '__main__':
    main()