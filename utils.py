import torch
import torch.nn as nn
from torch.autograd import Variable
import config


def powerset(s):
    """
    生成集合 s 的所有子集（幂集）。
    复刻自 notebook 中的 powerset 函数。
    """
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


class FNN_asig(nn.Module):
    """
    FNN with Structured Layer.
    复刻自 notebook 中的 FNN_asig 类 (Cell 3c2c19db).
    """

    def __init__(self):
        """FNN Builder."""
        super(FNN_asig, self).__init__()

        # 对应 notebook: 
        # self.layer1 = nn.Sequential(
        #     nn.Linear(d_c, 10, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(10, m+1)
        # )
        # 用于预测 nuisance parameter theta(x) (即代码中的 beta)
        self.layer1 = nn.Sequential(
            nn.Linear(config.d_c, 10, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(10, config.m + 1)
        )

        self.siglayer = nn.Sigmoid()

        # 对应 notebook: self.layer3 = nn.Linear(1, 1, bias=False)
        # 用于对 sigmoid 的输出进行缩放 (即 parameter c_est)
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
    复刻自 notebook 中的 calculate_mse 函数。
    """
    cnt = 0
    total_loss = 0

    # 确保在计算误差时不需要计算梯度
    # notebook 中虽然没有显式写 net.eval()，但为了准确性通常需要
    # 这里为了保持严格一致，我们只在逻辑上复刻

    for data in loader:
        inputs, labels = data

        # 处理设备 (GPU/CPU)
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        # Notebook 中使用了 Variable，但在新版 PyTorch 中 Tensor 即可
        # 为保持兼容性，我们直接使用 Tensor
        # inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)

        cnt += labels.size(0)
        # 对应 notebook: total_loss += sum((outputs-labels)**2)
        total_loss += torch.sum((outputs - labels) ** 2).item()

    return total_loss / float(cnt)