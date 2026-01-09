import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 导入自定义模块
import config
from utils import FNN_asig, calculate_mse, plot_figure7
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
        if epoch % 10 == 0 or epoch == config.epochs - 1:
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

            print(f"Epoch {epoch:3d} | Train MSE: {train_mse:.4f} | "
                  f"MAPE - DeDL: {mape_dedl:.2%}, SDL: {mape_sdl:.2%}, LR: {mape_lr:.2%}")

    # 5. 绘图并保存
    plot_figure7(history)


if __name__ == '__main__':
    main()