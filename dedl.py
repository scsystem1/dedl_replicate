import numpy as np
import torch
from numpy.linalg import inv
import config


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_network_params(net, x_input):
    """
    从神经网络中获取中间层输出 beta (theta) 和参数 c。

    Args:
        net: 训练好的 FNN_asig 模型
        x_input: 输入特征 X (numpy array), shape [n, d_c]

    Returns:
        beta: 神经网络 layer1 的输出, shape [n, m+1]
        c_est: 神经网络 layer3 的权重, scalar
    """
    net.eval()
    with torch.no_grad():
        x_tensor = torch.Tensor(x_input).to(config.device)
        beta = net.layer1(x_tensor).cpu().numpy()
        c_est = net.layer3.weight.data.cpu().numpy()[0, 0]
    return beta, c_est


def calculate_gradient(beta, c, t):
    """
    计算 Link Function G 对参数 (beta, c) 的梯度。

    G = c * sigmoid(beta . t)

    Derivatives:
    dG/d_beta = c * sigmoid(u) * (1 - sigmoid(u)) * t
    dG/d_c    = sigmoid(u)
    where u = beta . t

    Returns:
        gradient vector of shape [m+2] (beta部分 m+1, c部分 1)
    """
    u = np.dot(beta, t)
    sig_u = sigmoid(u)

    # 梯度计算，与 notebook 保持一致
    d_beta = c * np.exp(-u) / (np.exp(-u) + 1) ** 2 * t
    d_c = np.array([sig_u])

    return np.concatenate((d_beta, d_c))


def get_debiased_prediction(net, x_est, t_est, y_est, t_target, params):
    """
    计算给定目标 Treatment t_target 下的 Plug-in 预测值和 DeDL 去偏预测值。

    Args:
        net: 训练好的模型
        x_est: 评估用的特征 X (n_est, d_c)
        t_est: 评估用的观测 Treatment T (n_est, m+1)
        y_est: 评估用的观测 Outcome Y (n_est,)
        t_target: 目标 Treatment 向量 (m+1,)
        params: 包含 t_combo_obs, t_dist_obs 等信息

    Returns:
        y_pred_mean: SDL (Plug-in) 估计的均值
        y_debiased_mean: DeDL (Debiased) 估计的均值
    """
    # 1. 获取网络参数
    beta_est, c_est = get_network_params(net, x_est)

    n_samples = x_est.shape[0]
    m = config.m

    # 获取模型在观测数据上的预测值 (用于计算残差)
    net.eval()
    with torch.no_grad():
        # 构造输入 [x, t_obs]
        input_obs = np.hstack((x_est, t_est))
        input_tensor = torch.Tensor(input_obs).to(config.device)
        # notebook 中的 pred_y_loss
        pred_y_obs = net(input_tensor).cpu().numpy()

    # 获取模型在目标 Treatment 上的预测值 (SDL Estimator)
    # 构造输入 [x, t_target]
    t_target_expanded = np.tile(t_target, (n_samples, 1))
    input_target = np.hstack((x_est, t_target_expanded))
    with torch.no_grad():
        input_target_tensor = torch.Tensor(input_target).to(config.device)
        # notebook 中的 pred_y
        pred_y_target = net(input_target_tensor).cpu().numpy()

    # 2. 核心循环：计算 Lambda, Lambda_inv, G_theta
    # 这部分逻辑严格对应 Notebook 中的循环

    t_combo_obs = params['t_combo_obs']
    t_dist_obs = params['t_dist_obs']

    lambda_inv_list = []
    G_theta_target_list = []  # 目标 treatment 的梯度
    G_theta_obs_list = []  # 观测 treatment 的梯度

    for i in range(n_samples):
        beta_i = beta_est[i]

        # A. 计算 Lambda 矩阵
        # Lambda = E[2 * G' G'^T]
        lambda_mat = np.zeros((m + 2, m + 2))

        for k in range(len(t_combo_obs)):
            t_k = t_combo_obs[k]
            prob_k = t_dist_obs[k]

            # 计算梯度 G_prime
            G_prime = calculate_gradient(beta_i, c_est, t_k)

            # 累加外积
            lambda_mat += prob_k * 2 * np.outer(G_prime, G_prime)

        # B. 计算 Lambda 的逆 (加正则项)
        try:
            # inv(lambda_ + reg_term*np.eye(m+2))
            lam_inv = inv(lambda_mat + config.reg_term * np.eye(m + 2))
        except np.linalg.LinAlgError:
            print(f'Singular matrix at index {i}, utilizing identity fallback.')
            lam_inv = np.eye(m + 2)
        lambda_inv_list.append(lam_inv)

        # C. 计算当前样本在 目标 Treatment 下的梯度
        G_target = calculate_gradient(beta_i, c_est, t_target)
        G_theta_target_list.append(G_target)

        # D. 计算当前样本在 观测 Treatment 下的梯度
        G_obs = calculate_gradient(beta_i, c_est, t_est[i])
        G_theta_obs_list.append(G_obs)

    # 3. 计算去偏项 correction
    # lambda_inv_loss_prime = 2 * (pred - y) * lambda_inv . G_theta_obs
    correction_terms = []
    for i in range(n_samples):
        resid = pred_y_obs[i] - y_est[i]
        term = 2 * resid * np.dot(lambda_inv_list[i], G_theta_obs_list[i])
        correction_terms.append(term)

    # 4. 计算 DeDL 预测值
    # pred_debiased = pred_target - G_theta_target . correction_term
    pred_y_debiased = []
    for i in range(n_samples):
        correction = np.dot(G_theta_target_list[i], correction_terms[i])
        pred_y_debiased.append(pred_y_target[i] - correction)

    pred_y_debiased = np.array(pred_y_debiased)

    # 返回均值作为 ATE 的组成部分 (E[Y|T=t])
    return np.mean(pred_y_target), np.mean(pred_y_debiased)


def calculate_ate(net, x_est, t_est, y_est, t_combo, params):
    """
    计算所有 Treatment 组合的平均 MAPE。

    Returns:
        mape_sdl, mape_dedl, mape_lr
    """
    # 预先训练一个 LR 模型作为 Baseline (与 notebook 一致，每次评估时训练)
    # LR: y ~ [x, t] (无交互项或简单的线性交互)
    # Notebook 中: model_LR = sm.OLS(...).fit()
    # 我们使用 statsmodels
    import statsmodels.api as sm

    X_LR_train = np.hstack((x_est, t_est))
    model_LR = sm.OLS(y_est, X_LR_train).fit()
    coef_LR = model_LR.params

    mape_sdl_list = []
    mape_dedl_list = []
    mape_lr_list = []

    # 真实参数用于计算 Ground Truth
    coef_true = params['coef']
    c_true = params['c_true']
    d_true = params['d_true']

    # 基准 Treatment (全0，除了 intercept 位 t[0]=1)
    t_base = params['t_combo'][0]

    # 1. 计算 Base Case 的估计值
    base_sdl, base_dedl = get_debiased_prediction(net, x_est, t_est, y_est, t_base, params)

    # 计算 Base Case 的 LR 预测
    X_LR_base = np.hstack((x_est, np.tile(t_base, (config.n_est, 1))))
    base_lr = np.mean(np.dot(X_LR_base, coef_LR))

    # 计算 Base Case 的 Ground Truth
    # generate_y_true 的逻辑: y = c / (1+exp(-((x.coef)**3).t))
    term_base = np.sum((np.dot(x_est, coef_true)) ** 3 * t_base, axis=1)
    y_true_base = np.mean(c_true / (1 + np.exp(-term_base)) + d_true)

    # 2. 遍历所有 Treatment (除了 Base)
    for t_star in t_combo:
        if np.array_equal(t_star, t_base):
            continue

        # 计算 Target Case 的估计值
        target_sdl, target_dedl = get_debiased_prediction(net, x_est, t_est, y_est, t_star, params)

        # 计算 Target Case 的 LR 预测
        X_LR_target = np.hstack((x_est, np.tile(t_star, (config.n_est, 1))))
        target_lr = np.mean(np.dot(X_LR_target, coef_LR))

        # 计算 Target Case 的 Ground Truth
        term_target = np.sum((np.dot(x_est, coef_true)) ** 3 * t_star, axis=1)
        y_true_target = np.mean(c_true / (1 + np.exp(-term_target)) + d_true)

        # 计算 ATE
        ate_true = y_true_target - y_true_base
        ate_sdl = target_sdl - base_sdl
        ate_dedl = target_dedl - base_dedl
        ate_lr = target_lr - base_lr

        # 计算 MAPE (避免除零)
        if abs(ate_true) > 1e-6:
            mape_sdl_list.append(abs(ate_sdl - ate_true) / abs(ate_true))
            mape_dedl_list.append(abs(ate_dedl - ate_true) / abs(ate_true))
            mape_lr_list.append(abs(ate_lr - ate_true) / abs(ate_true))

    return np.mean(mape_sdl_list), np.mean(mape_dedl_list), np.mean(mape_lr_list)