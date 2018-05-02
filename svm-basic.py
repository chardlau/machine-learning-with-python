import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# 这是测试数据
data = {
    '+': [
        [1, 7],
        [2, 8],
        [3, 8],
        [2, 6.5]
    ],
    '-': [
        [5, 1],
        [6, -1],
        [7, 3]
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

X, y, c, c0 = parseXYC(data)

# 求解“F = (1/2)*x.T*H*x + c*x + c0”函数的最小情况下x的取值
# 约束条件为“x >= 0”且“y*x = 0”

# 计算H矩阵，根据样本数目应是一个len(X)xlen(X)的矩阵
H = np.array([y[i] * y[j] * np.dot(X[i], X[j]) for i in range(len(X)) for j in range(len(X))]).reshape(len(X), len(X))

# 定义二项规划方程fun及其雅各比方程jac
def fun(x, sign=1.):
    return sign * (0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)
def jac(x, sign=1.):
    return sign * (np.dot(x.T, H) + c)

# 定义等式约束条件方程feq及其雅各比方程jeq
def feq(x):
    return np.dot(y, x)
def jeq(x):
    return np.array(y)

# 生成相关参数
diff = 1e-16
bounds = [(0, None) for _ in range(len(y))] # x >= 0
constraints = [{ 'type': 'eq', 'fun': feq, 'jac': jeq }]# y*x = 0
options = { 'ftol': diff, 'disp': True }
guess = np.array([0 for _ in range(len(X))])
res_cons = optimize.minimize(fun, guess, method='SLSQP', jac=jac, bounds=bounds, constraints=constraints, options=options)
alpha = [ 0 if abs(x - 0) <= diff else x for x in res_cons.x ]
print('raw alpha: ', res_cons.x)
print('fmt alpha: ', alpha)
print('check y*alpha: ', 'is 0'if (abs(np.dot(y, res_cons.x) - 0) < diff ) else 'is not 0')

# 计算w = sum(xi*yi*Xi)
w = np.sum([ np.array([0, 0]) if alpha[i] == 0 else (alpha[i] * y[i] * X[i]) for i in range(len(alpha))], axis=0)
print('w: ', w)

# 计算b，对support vector有：yi(w*xi + b) = 1，既有：b = 1/yi - w*xi
B = [( 0 if alpha[i] == 0 else ( 1 / y[i] - np.dot(w, X[i]) ) ) for i in range(len(alpha))]
B = list(filter(lambda x: x != 0, B))
b = 0 if len(B) <= 0 else B[0]
print('b: ', b)

limit = 11
plt.xlim(-2, limit)
plt.ylim(-2, limit)
# 绘制数据点
[plt.scatter(X[i][0],X[i][1], s=100, color=('r' if y[i] > 0 else 'y')) for i in range(len(X))]
# 绘制分割超平面L： wx + b = 0
plt.plot([i for i in range(limit)], [(-b - w[0]*i)/w[1] for i in range(limit)])
# 绘制上下边： wx + b = 1/-1
plt.plot([i for i in range(limit)], [(1-b - w[0]*i)/w[1] for i in range(limit)])
plt.plot([i for i in range(limit)], [(-1-b - w[0]*i)/w[1] for i in range(limit)])
plt.show()


