import torch

m = 4               # 实验数量 (number of experiments)
d_c = 10            # 用户特征维度 (number of features)
n_train = 2000      # 训练样本数量 (int(m * 500) in notebook)
n_est = 2000        # 推断/评估样本数量

lr = 0.05           # 学习率 (Learning rate)
wd = 5e-4           # 权重衰减 (Weight decay / L2 regularization)
reg_term = 0.0005   # 矩阵求逆时的正则化项 (Regularization for matrix inversion)
reg_loss = 0        # 损失函数中的 L1 正则化系数 (L1 regularization in loss)

epochs = 250        # 训练轮数
batch_size_train = 1000
batch_size_test = 1000

# --------------------------
# 系统配置 (System Config)
# --------------------------
seed = 42           # 随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")