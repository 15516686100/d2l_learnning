import os
import pandas as pd
import torch

# 创建人工数据集
path = os.getcwd()              # 获取当前工作路径
os.makedirs(os.path.join(path, 'data'), exist_ok=True)    # 在当前工作路径下创建data文件夹
data_file = os.path.join(path, 'data', 'house_tiny.csv')  # 在data文件下创建house_tiny.csv文件

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')                     # 列名
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取文件内容
data = pd.read_csv(data_file)
# print(data)

# 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())                # 将inputs中的NaN值填充为均值
# print(inputs)
'''
对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。
由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 
缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
'''
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)

# 将文件中的数据转换为张量
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, '\n', y)
