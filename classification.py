import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd

# 读取乳癌统计数据
df = pd.read_csv('./dataset/breast-cancer-wisconsin.data')
# 处理问号
df.replace('?', -99999, inplace=True)
# 因为ID字段与分类无关，所以去除他先，稍后我们看一下它的影响
df.drop(['id'], 1, inplace=True)
df = df.astype(float)

# 生成数据集
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# 构建模型与训练
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# 计算精确度
accuracy = clf.score(X_test, y_test)
print('Accuracy: ', accuracy)

# 预测我们自己构造的数据属于哪个类型
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[2,3,4,4,1,2,3,4,1]])
prediction = clf.predict(example_measures)
print('Predict resuct: ', prediction)

