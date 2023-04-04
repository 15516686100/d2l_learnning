import torch

# 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
# print(x+y, x-y, x*y, x/y, x**y)          # tensor(5.) tensor(1.) tensor(6.) tensor(1.5000) tensor(9.)

# 向量
z = torch.arange(4)
# print(f'z={z:}')                         # z=tensor([0, 1, 2, 3])
# print(z[3])                              # 索引:tensor(3)
# print(len(z))                            # 长度:4
# print(z.shape)                           # 形状:torch.Size([4])

# 矩阵
A = torch.arange(0, 16, 3).reshape(2, 3)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(A)                                   # tensor([[ 0,  3,  6],[ 9, 12, 15]])
# print(A.T)                                 # tensor([[ 0,  9],[ 3, 12],[ 6, 15]])
# print(B)
# print(B == B.T)                            # tensor([[True, True, True],[True, True, True],[True, True, True]])

# 张量
X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
# print(X)
Y = X.clone()
# print(Y)
# print(X == Y)
# print(X + Y)                                # 各个位置相加相乘
# print(X * Y)                                # 哈德曼积: 各个位置元素相乘，不改变形状
m = 2
# print(m + X, (m * X).shape)                 # torch.Size([2, 3, 4])

# 降维
# print(X.sum())                                # 求和:tensor(276)
# 对单独某一维度求和
X_sum_axis0 = X.sum(axis=0)
X_sum_axis1 = X.sum(axis=1)
X_sum_axis2 = X.sum(axis=2)
# print(f'X_sum_axis0:\n{X_sum_axis0}')
# print(X_sum_axis0.shape)                      # torch.Size([3, 4])
# print(f'X_sum_axis1:\n{X_sum_axis1}')
# print(X_sum_axis1.shape)
# print(f'X_sum_axis2:\n{X_sum_axis2}')
# print(X_sum_axis2.shape)

# 求均值
# print(X.mean())                              # tensor(11.5000)
# print(X.sum() / X.numel())                   # tensor(11.5000)
# print(X.mean(axis=0))
# print(X.sum(axis=0) / X.shape[0])

# 点积(Dot Product)
a = torch.arange(4)
b = torch.ones_like(a)
# print(a)                                       # tensor([0, 1, 2, 3])
# print(b)                                       # tensor([1, 1, 1, 1])
# print(torch.dot(a, b))                         # tensor(6)
# print(torch.sum(a * b))                        # tensor(6)

# 矩阵向量积
C = torch.arange(20).reshape(5, 4)
# print(C)
# print(torch.mv(C, a))

# 矩阵乘法
D = torch.arange(12).reshape(4, 3)
# print(torch.mm(C, D).shape)                   # torch.Size([5, 3])

# 范数
u = torch.tensor([3.0, -4.0])
# print(torch.norm(u))                           # L2范数，欧几里得距离:tensor(5.)
# print(torch.abs(u).sum())                      # L1范数，向量元素的绝对值之和:tensor(7.)
