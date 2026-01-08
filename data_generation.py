import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import config
from utils import powerset  # 假设下一文件 utils.py 中会实现此函数


def generate_x(d_c, n):
    """生成 d_c 维的均匀分布特征数据"""
    return np.random.uniform(0, 1, [n, d_c])


def generate_t(t_combo, t_dist, n):
    """根据给定的分布采样 treatment 组合"""
    # 从 t_combo 中根据 t_dist 概率随机选择 n 个索引
    indices = np.random.choice(np.shape(t_dist)[0], n, p=t_dist)
    return t_combo[indices, :]


def generate_y_true(coef, c, d, x, t, n):
    """
    生成 Outcome Y (包含噪声)。

    关键细节:
    原代码(Cell 6036fb85)使用了 ((x.dot(coef))**3).dot(t)
    作为 sigmoid 的输入，制造了非线性关系，这正是 Figure 7 实验的设定。
    """
    y = np.zeros(n)
    # 噪声范围 [-0.05, 0.05]
    y_error = 0.05 * np.random.uniform(-1, 1, n)

    for i in range(n):
        # 注意这里的三次方 (**3)，复刻原代码逻辑
        # x[i]: (d_c,), coef: (d_c, m+1) -> x.dot(coef): (m+1,)
        term = ((x[i].dot(coef)) ** 3).dot(t[i])
        y[i] = c / (1 + np.exp(-term)) + d + y_error[i]

    return y, y_error


def get_data():
    """
    生成训练数据和用于评估 Ground Truth 的参数
    """
    # 设置随机种子以保证结果可复现
    np.random.seed(config.seed)

    m = config.m
    d_c = config.d_c
    n_train = config.n_train

    # 1. 定义所有可能的 Treatment 组合 (2^m 个)
    all_combo_indices = list(powerset(list(np.arange(1, m + 1))))
    t_combo = []
    # 基础组合 (全0) 也需要包含吗？
    # 原代码 powerset 生成非空子集，但也手动处理了 Base
    # 下面的逻辑与原代码一致：
    for indices in all_combo_indices:
        t = np.zeros(m + 1)
        t[0] = 1  # Intercept bit (Bias term)
        for idx in indices:
            t[idx] = 1
        t_combo.append(t)

    # 手动添加 Base case (仅 Intercept 为 1) 到列表开头或单独处理
    # 原代码中 t_combo 包含了所有 2^m 种情况
    # 这里我们确保 t_combo 完整包含所有情况
    # 原代码 powerset 生成的是 1..m 的子集，不含空集，所以 t_combo 实际上有 2^m-1 个?
    # 不，原代码 t[0]=1 始终存在。空集对应只有 t[0]=1。
    # 让我们确保 t_combo 包含全 0 (除 t[0] 外) 的情况
    t_base = np.zeros(m + 1)
    t_base[0] = 1
    # 检查 t_base 是否已在 t_combo 中 (powerset 不含空集，所以 t_base 需要手动加)
    t_combo = [t_base] + t_combo
    t_combo = np.array(t_combo, dtype=np.int16)

    # 2. 定义观测到的 Treatment 组合 (t_combo_obs)
    # 原代码逻辑：Base + Single Experiments + 组合[1,2,3]
    t_combo_obs = []

    # Base + Single Experiments
    for i in range(m + 1):
        t = np.zeros(m + 1)
        t[0] = 1
        t[i] = 1  # 当 i=0 时，t=[1,0,0...] (Base); 当 i=1 时, t=[1,1,0...]
        t_combo_obs.append(t)

    # 添加特定的组合: Exps 1, 2, 3 同时发生
    t_complex = np.zeros(m + 1)
    t_complex[[0, 1, 2, 3]] = 1  # 原代码: t[[0,1,2]] = 1 (注意索引偏移，原代码索引从1开始?)
    # 原代码: indices 1,2,3 对应 t 的 1,2,3 位 (0位是intercept)
    # 原代码: t[[0,1,2]] = 1 实际上设置了 intercept, exp1, exp2。
    # 让我们再看一眼原代码:
    # "t_combo_obs.append(t); t = np.zeros(m+1); t[[0,1,2]] = 1"
    # 是的，这里稍微有点歧义。如果 m=4，t 长度为 5。
    # t[[0,1,2]] = 1 意味着 intercept, exp1, exp2 被激活。
    # 我们严格遵照原代码 t[[0,1,2]] = 1
    t_complex = np.zeros(m + 1)
    t_complex[[0, 1, 2]] = 1
    t_combo_obs.append(t_complex)

    t_combo_obs = np.array(t_combo_obs, dtype=np.int16)

    # 观测分布 (均匀)
    # 注意: t_combo_obs 的长度是 (m+1) + 1 = m+2
    t_dist_obs = (1 / (m + 2)) * np.ones(m + 2)

    # 3. 生成真实参数
    # coef: d_c x (m+1)
    coef = np.random.uniform(-0.5, 0.5, [d_c, m + 1])
    c_true = np.random.uniform(10, 20)
    d_true = 0

    # 4. 生成训练数据 (Training Data)
    samples_x = generate_x(d_c, n_train)
    samples_t = generate_t(t_combo_obs, t_dist_obs, n_train)
    samples_y, _ = generate_y_true(coef, c_true, d_true, samples_x, samples_t, n_train)

    # 转换为 PyTorch DataLoader
    # 输入: [X, T]
    x_input = np.hstack((samples_x, samples_t))

    # 使用 config 中的 dtype (float32)
    dataset = TensorDataset(
        torch.tensor(x_input, dtype=torch.float32),
        torch.tensor(samples_y, dtype=torch.float32)
    )

    # 注意: batch_size 和 shuffle 设置
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train,
        shuffle=True
    )

    # 5. 准备用于推断/评估的数据 (Inference Meta-data)
    # 我们不预先生成所有 inference 的 y，因为那是动态评估的
    # 我们返回生成 y 所需的参数，以便在 dedl.py 中计算 Ground Truth
    data_info = {
        'coef': coef,
        'c_true': c_true,
        'd_true': d_true,
        't_combo': t_combo,  # 所有可能的组合 (用于评估所有情况的 ATE)
        't_combo_obs': t_combo_obs,  # 观测到的组合 (用于计算 Lambda)
        't_dist_obs': t_dist_obs
    }

    return train_loader, data_info