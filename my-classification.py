import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')

# 计算欧几里得距离
def euclidean_distance(x, y):
    if len(x) != len(y):
        warnings.warn('Input error')
    return sqrt( sum( [(x[i] - y[i])**2 for i in range(0, len(x))] ) )
#print(euclidean_distance([1,2,3], [2,4,5]))

# NumPy计算欧几里得距离
#print(np.linalg.norm(np.array([1,2,3]) - np.array([2,4,5])))

# 测试数据
#dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
#new_features = [5,7]

# KNN实现
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K less than total voting groups')
    # 计算距离
    distances = []
    for group in data:
        for features in data[group]:
            #distance = euclidean_distance(features, predict)
            distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([distance, group])
    # 排序后取前k项数据类别构成新数组
    votes = [i[1] for i in sorted(distances)[:k]]
    # 统计数组中频数最高的类别
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

# 测试KNN函数
#result = k_nearest_neighbors(dataset, new_features, k=3)
#print(result)

# 绘制数据
#[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1])
#plt.show()


# 读取乳癌统计数据
df = pd.read_csv('./dataset/breast-cancer-wisconsin.data')
# 处理问号
df.replace('?', -99999, inplace=True)
# id字段不应该当成一个统计特征字段，因此去除该列的内容
df.drop(['id'], 1, inplace=True)
# 源数据有部分数据是字符串，如'1'，这对我们的模型有影响，所以整理一下类型
# 我们只接受列表作为输入
full_data = df.astype(float).values.tolist()
random.shuffle(full_data) # 洗乱数据


# 生成训练数据集和统计数据集
test_size = 0.2
train_set = {2:[], 4:[]} # 训练集，占80%
test_set = {2:[], 4:[]} # 统计集，占20%
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

# 利用上述KNN函数统计测试数据的准确性
correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
# 打印结果
print('correct: ', correct)
print('total: ', total)
print('Accuracy: ', correct/total)