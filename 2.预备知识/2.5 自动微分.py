import torch

"""自动求导简单实现"""
x = torch.arange(4.0, requires_grad=True)
print(x, x.grad)                           # tensor([0., 1., 2., 3.], requires_grad=True) None
y = 2 * torch.dot(x, x)                    # 等价于y=2x²
print(y)
y.backward()                               # 调用反向传播函数
print(x.grad)                              # y关于x的导数是4x,x的梯度:tensor([ 0.,  4.,  8., 12.])

x.grad.zero_()                             # 梯度清零
print(x.grad)                              # tensor([0., 0., 0., 0.])
z = 2 * x ** 3 - 4 * x                     # z关于x的导数2为6x-4
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
z.sum().backward()                         # 等价于z.backward(torch.ones(len(x)))
# z.backward(torch.ones(len(x)))
print(x.grad)                              # tensor([-4.,  2., 20., 50.])

"""分离计算"""
x.grad.zero_()
y = x * x
u = y.detach()                             # 将u视作一个常数处理
print(u)                                   # tensor([0., 1., 4., 9.])
z = u * x
z.sum().backward()                         # z关于x的梯度是u，若不做detach处理，z=x³，梯度为3x²
print(x.grad)                              # tensor([0., 1., 4., 9.])
x.grad.zero_()
m = x * x * x
m.sum().backward()
print(x.grad)                              # tensor([ 0.,  3., 12., 27.])


"""控制流的梯度计算"""
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
d.backward()
print(d)
print(a.grad)
print(a.grad == d / a)