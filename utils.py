import torch
import torch.nn as nn
from torch.autograd import Variable
import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns


def plot_figure7(history, save_path="Figure7_Replication.pdf"):
    epochs = history['epoch']
    mse = history['train_mse']

    sdl_mape = np.array(history['mape_sdl']) * 100
    dedl_mape = np.array(history['mape_dedl']) * 100
    lr_mape = np.array(history['mape_lr']) * 100

    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.linewidth': 1.0,
        'savefig.dpi': 600,
        'figure.dpi': 600
    })

    sns.set_theme(style="white", font="Times New Roman")

    # --- 左轴: MSE ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_mse = '#D62728'
    ax1.set_xlabel('Training Epoch', fontsize=16)
    ax1.set_ylabel('Training MSE', color=color_mse, fontsize=16)

    # 绘制 MSE 线
    line1, = ax1.plot(epochs, mse, color=color_mse,
                      linewidth=1.5, alpha=0.8, label='Training MSE')

    ax1.tick_params(axis='y', labelcolor=color_mse, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_ylim(bottom=0)

    # --- 右轴: Estimation MAPE (不同方法) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Estimation MAPE (%)', color='black', fontsize=16)
    # DeDL: 实线，深蓝色/黑色
    line2, = ax2.plot(epochs, dedl_mape, color='#1F77B4', linestyle='-',
                      linewidth=2.5, label='DeDL MAPE (Ours)')

    # SDL: 虚线，灰色
    line3, = ax2.plot(epochs, sdl_mape, color='gray', linestyle='--',
                      linewidth=2, label='SDL MAPE')

    # LR: 点线，灰色
    line4, = ax2.plot(epochs, lr_mape, color='gray', linestyle=':',
                      linewidth=2, label='LR MAPE')

    ax2.tick_params(axis='y', labelcolor='black', labelsize=14, direction='in')

    ax2.yaxis.set_tick_params(which='both', direction='in')
    top_lim = max(np.max(sdl_mape), np.max(dedl_mape), np.max(lr_mape)) * 1.3
    ax2.set_ylim(0, top_lim)

    # --- 图例与布局 ---
    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, frameon=False, fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    # plt.show() # 如果在服务器运行，可以注释掉

def powerset(s):
    """
    生成集合 s 的所有子集
    """
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


class FNN_asig(nn.Module):
    """
    FNN with Structured Layer.
    """

    def __init__(self):
        """FNN Builder."""
        super(FNN_asig, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(config.d_c, 10, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(10, config.m + 1)
        )

        self.siglayer = nn.Sigmoid()
        self.layer3 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        """Perform forward."""
        # 输入 x 包含了 Features 和 Treatments
        # x[:, 0:d_c] 是 Features X
        # x[:, d_c:]  是 Treatments T

        # 1. 计算 theta(x) / beta
        b = self.layer1(x[:, 0:config.d_c])

        # 2. 计算结构化部分的线性组合 u = theta(x) . T
        # 对应 notebook: u = torch.sum(b*x[:, d_c:], 1)
        u = torch.sum(b * x[:, config.d_c:], 1)

        # 3. Sigmoid 变换
        u = self.siglayer(u)

        # 4. 线性缩放 (乘 c)
        u = u.unsqueeze(1)
        u = self.layer3(u)

        return torch.reshape(u, (-1,))


def calculate_mse(loader, net):
    """
    Calculate Mean Squared Error.
    """
    cnt = 0
    total_loss = 0

    for data in loader:
        inputs, labels = data

        # 处理设备 (GPU/CPU)
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        outputs = net(inputs)

        cnt += labels.size(0)
        total_loss += torch.sum((outputs - labels) ** 2).item()

    return total_loss / float(cnt)