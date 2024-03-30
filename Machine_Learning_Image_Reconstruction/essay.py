import numpy as np
'''
基础MRI模型
y = EX + n
'''
# 定义图像空间信号 x
x = np.array([3, 1])  # 可以更改为任何值

# 定义傅立叶变换矩阵 F
F = 1/np.sqrt(2) * np.array([[1, 1],
                             [1, -1]])

# 定义采样掩膜 M
M = np.array([[1, 0],
              [0, 0]])

# 计算编码矩阵 E = MF
E = M.dot(F)

# 应用编码矩阵 E 到 x 上，得到 y
y = E.dot(x)

print("编码矩阵 E:\n", E)
print("k空间数据 y:\n", y)
