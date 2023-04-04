import torch
import numpy as np

# 创建张量
x = torch.arange(12)
# print(x)              # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# print(x.shape)        # torch.Size([12])
# print(x.numel())      # 12

# 改变张量的形状
X = x.reshape(3, 4)
# print(X)              # tensor([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]])

# 创建全零张量
zeros = torch.zeros(2, 3, 4)
# print(zeros)          # tensor([[[0., 0., 0., 0.],[0., 0., 0., 0.],[0., 0., 0., 0.]],[[0., 0., 0., 0.],[0., 0., 0., 0.],[0., 0., 0., 0.]]])

# 创建全一张量
ones = torch.ones(3, 4)
# print(ones)             # tensor([[1., 1., 1., 1.],[1., 1., 1., 1.],[1., 1., 1., 1.]])

# 创建指定全填充张量
fulls = torch.full((2, 3), 2)
# print(fulls)              # tensor([[2, 2, 2],[2, 2, 2]])

# 从numpy中创建张量并保存到GPU中
arr = np.array([[1, 2, 3], [4, 5, 6]])
gpu_arr = torch.tensor(arr, device='cuda')
cpu_arr = torch.tensor(arr)
# print(gpu_arr)            # tensor([[1, 2, 3],[4, 5, 6]], device='cuda:0', dtype=torch.float64)
# print(cpu_arr)            # tensor([[1, 2, 3],[4, 5, 6]], dtype=torch.float64)

# 创建高斯正态分布张量
normals = torch.randn(2, 3)                  # 形状2×3的高斯分布张量:均值为0，标准差为1
normals_like = torch.randn_like(ones)        # 与上文全1张量形状相同的高斯分布：3×4
# print(normals)                             # tensor([[ 1.9111, -0.8631,  0.0389],[ 1.4233,  0.2239, -0.0721]])
# print(normals_like)                        # tensor([[ 1.5444, -0.6738, -0.6879,  1.2293],[-0.1435, -0.2222, -0.7740,  0.4837],[-0.1187, -0.1076,  0.1128,  0.5511]])

# 张量运算
x1 = torch.tensor([1.0, 2, 4, 8])
y1 = torch.tensor([2, 2, 2, 2])
# print(x1 + y1)                              # tensor([ 3.,  4.,  6., 10.])
# print(x1 - y1)                              # tensor([-1.,  0.,  2.,  6.])
# print(x1 * y1)                              # tensor([ 2.,  4.,  8., 16.])
# print(x1 / y1)                              # tensor([0.5000, 1.0000, 2.0000, 4.0000])
# print(x1 ** y1)                             # tensor([ 1.,  4., 16., 64.])

# 张量连接
X1 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y1 = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
XY = torch.cat((X1, Y1), dim=0)            # 在第0维，行上连接
YX = torch.cat((X1, Y1), dim=1)            # 在第1维，列上连接
# print(XY)
# print(YX)

# 对张量中所有元素求和
# print(X1.sum())                            # tensor(66.)

# 广播机制
'''
两个张量之间进行按位运算，一般需要相同的形状，在默认情况下，即使形状不同，仍然可以通过调用
广播机制(broadcasting mechanism)来执行按位操作，广播机制的工作方式如下:
  1.通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状;
  2.对生成的数组执行按元素操作
'''
a = torch.arange(3).reshape((3, 1))          # a是3×1的矩阵
b = torch.arange(2).reshape((1, 2))          # b是1×2矩阵
# print(a, '\n', b)
# print(a + b)                                 # 广播机制后，a复制列3×2，b复制行3×2

