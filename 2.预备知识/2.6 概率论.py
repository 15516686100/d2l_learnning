import torch
from torch.distributions import multinomial
from matplotlib import pyplot as plt

fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(1, fair_probs).sample())
print(multinomial.Multinomial(10, fair_probs).sample())

# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)  # 相对频率作为估计值

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)


def set_fig_size(fig_size):
    """设置图像大小"""
    plt.rcParams['figure.figsize'] = fig_size


def plt_show(fig_size):
    set_fig_size(fig_size)
    for i in range(6):
        plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
        plt.axhline(y=0.167, color='black', linestyle='dashed')
        plt.gca().set_xlabel('Groups of experiments')
        plt.gca().set_ylabel('Estimated probability')
        plt.legend()
    plt.show()

plt_show((8.0, 6.0))
