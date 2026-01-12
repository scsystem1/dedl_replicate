import torch

m = 4               # 实验数量 (number of experiments)
d_c = 10            # 用户特征维度 (number of features)
n_train = 2000      # 训练样本数量 (int(m * 500) in notebook)
n_est = 2000        # 推断/评估样本数量

lr = 0.05
wd = 5e-4
reg_term = 0.0005
reg_loss = 0

epochs = 400
batch_size_train = 1000
batch_size_test = 1000
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")