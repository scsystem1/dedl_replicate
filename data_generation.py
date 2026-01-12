import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import config
from utils import powerset


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
    """
    y = np.zeros(n)
    # 噪声范围 [-0.05, 0.05]
    y_error = 0.05 * np.random.uniform(-1, 1, n)

    for i in range(n):
        term = ((x[i].dot(coef)) ** 3).dot(t[i])
        y[i] = c / (1 + np.exp(-term)) + d + y_error[i]

    return y, y_error


def get_data():
    """
    生成训练数据和用于评估 Ground Truth 的参数
    """
    np.random.seed(config.seed)

    m = config.m
    d_c = config.d_c
    n_train = config.n_train

    # 1. 定义所有可能的 Treatment 组合 (2^m 个)
    all_combo_indices = list(powerset(list(np.arange(1, m + 1))))
    t_combo = []
    for indices in all_combo_indices:
        t = np.zeros(m + 1)
        t[0] = 1
        for idx in indices:
            t[idx] = 1
        t_combo.append(t)

    t_base = np.zeros(m + 1)
    t_base[0] = 1
    # 检查 t_base 是否已在 t_combo 中 (powerset 不含空集，所以 t_base 需要手动加)
    t_combo = [t_base] + t_combo
    t_combo = np.array(t_combo, dtype=np.int16)

    # 2. 定义观测到的 Treatment 组合 (t_combo_obs)
    t_combo_obs = []

    # Base + Single Experiments
    for i in range(m + 1):
        t = np.zeros(m + 1)
        t[0] = 1
        t[i] = 1
        t_combo_obs.append(t)

    t_complex = np.zeros(m + 1)
    observed = range(m)
    t_complex[observed] = 1
    t_combo_obs.append(t_complex)

    t_combo_obs = np.array(t_combo_obs, dtype=np.int16)

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