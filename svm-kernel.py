# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 这是测试数据
data = {
    '+': [
        [4, 3],
        [5, 2]
    ],
    '-': [
        [1, 7],
        [2, 8],
        [3, 8],
        [2, 6],
        [6, -1],
        [6, -2],
        [7, -2]
    ]
}


# 解析数据
def parseXYC(d):
    X = []
    y = []
    c = []
    for _, v in enumerate(d['+']):
        X.append(np.array(v))
        y.append(1)
        c.append(-1)
    for _, v in enumerate(d['-']):
        X.append(np.array(v))
        y.append(-1)
        c.append(-1)
    return X, y, c, 0


# 核函数: 多项式核函数c=0，d=2
# 选择依据与测试数据有关，这样的映射应该可以把数据分隔开
# 则kernel应该返回的内积如下：
def kernel(x1, x2):
    return np.dot(x1, x2)**2


# 映射函数
def to_z_space(ins):
    return np.array([ins[0]**2, math.sqrt(2)*ins[0]*ins[1], ins[1]**2])


X, y, c, c0 = parseXYC(data)

# 求解“F = (1/2)*x.T*H*x + c*x + c0”函数的最小情况下x的取值
# 约束条件为“x >= 0”且“y*x = 0”

# 计算H矩阵，根据样本数目应是一个len(X)xlen(X)的矩阵
H = np.array(
    [y[i] * y[j] * kernel(X[i], X[j]) for i in range(len(X)) for j in range(len(X))]
).reshape(len(X), len(X))


# 定义二项规划方程fun及其雅各比方程jac
def fun(x, sign=1.):
    return sign * (0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(c, x) + c0)


def jac(x, sign=1.):
    return sign * (np.dot(x.T, H) + c)


# 定义等式约束条件方程feq及其雅各比方程jeq
def feq(x):
    return np.dot(y, x)


def jeq(x):
    return np.array(y)


# 生成相关参数
diff = 1e-16
bounds = [(0, None) for _ in range(len(y))]  # x >= 0
constraints = [{'type': 'eq', 'fun': feq, 'jac': jeq}]  # y*x = 0
options = {'ftol': diff, 'disp': True}
guess = np.array([0 for _ in range(len(X))])
res_cons = optimize.minimize(
    fun, guess, method='SLSQP', jac=jac, bounds=bounds, constraints=constraints, options=options)
alpha = [0 if abs(x - 0) <= diff else x for x in res_cons.x]
print('raw alpha: ', res_cons.x)
print('fmt alpha: ', alpha)
print('check y*alpha: ', 'is 0'if (abs(np.dot(y, res_cons.x) - 0) < diff) else 'is not 0')

# 计算w = sum(alpha[i]*y[i]*Z[i])
w = np.sum([np.array([0, 0, 0]) if alpha[i] == 0 else (alpha[i] * y[i] * to_z_space(X[i])) for i in range(len(alpha))], axis=0)
print('w: ', w)

# 计算b，对support vector有：y[i](w*Z[i] + b) = 1，既有：b = 1/yi - w*xi
B = [(0 if alpha[i] == 0 else (1 / y[i] - np.dot(w, to_z_space(X[i])))) for i in range(len(alpha))]
B = list(filter(lambda t: t != 0, B))
b = 0 if len(B) <= 0 else B[0]
print('b: ', b)

fig = plt.figure()
ax1 = fig.add_subplot(221)
[ax1.scatter(X[i][0], X[i][1], s=30, color=('r' if y[i] > 0 else 'y')) for i in range(len(X))]

ax2 = fig.add_subplot(222, projection='3d')
[ax2.scatter(X[i][0]**2, math.sqrt(2)*X[i][0]*X[i][1], X[i][1]**2, s=30, color=('r' if y[i] > 0 else 'y')) for i in range(len(X))]

t1 = np.arange(-30, 60, 1)
t2 = np.arange(-30, 60, 1)
t1, t2 = np.meshgrid(t1, t2)
t3 = np.array([[(-b - w[0]*t1[j][i] - w[1]*t2[j][i]) / w[2] for i in range(len(t1))] for j in range(len(t1))])

ax3 = fig.add_subplot(223, projection='3d')
[ax3.scatter(X[i][0]**2, math.sqrt(2)*X[i][0]*X[i][1], X[i][1]**2, s=5, color=('r' if y[i] > 0 else 'y')) for i in range(len(X))]
ax3.plot_surface(t1, t2, t3, color=(.7, .7, .7, .3))

ax4 = fig.add_subplot(224, projection='3d')
[ax4.scatter(X[i][0]**2, math.sqrt(2)*X[i][0]*X[i][1], X[i][1]**2, s=5, color=('r' if y[i] > 0 else 'y')) for i in range(len(X))]
ax4.plot_surface(t1, t2, t3, color=(.7, .7, .7, .3))

plt.show()
