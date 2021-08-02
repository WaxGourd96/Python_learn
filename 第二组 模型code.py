import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


# 读取数据
df = pd.read_csv(r'C:\Users\PanYueyi\Desktop\学习\Python\bank.csv')
# 缺失值处理, 将unknown替换成出现最多的类别
cols = ['job', 'marital', 'education', 'default', 'loan']
for col in cols:
    counts = df[col].value_counts()
    df[col] = df[col].replace('unknown', counts.index[0])
    print('{}: unknown替换成{}'.format(col, counts.index[0]))
# 将类别的列编号
cols = ['job', 'marital', 'education', 'contact', 'poutcome']
for col in cols:
    counts = df[col].value_counts()
    value_map = dict((v, i) for i, v in enumerate(counts.index))
    df[col] = df[col].replace(value_map)
    print(col, value_map)
# 将yes no 替换成1 0
cols = ['default', 'housing', 'loan', 'y']
value_map = {'yes': 1, 'no': 0}
for col in cols:
    df[col] = df[col].replace(value_map)
    print(col, value_map)
# month
col = 'month'
value_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df[col] = df[col].replace(value_map)
print(col, value_map)
# day_of_week
col = 'day_of_week'
value_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
df[col] = df[col].replace(value_map)
print(col, value_map)
y = df['y'].values
x = df.drop('y', axis=1).values
# housing
col = 'housing'
value_map = {'yes': 1, 'no': 0, 'unknown': 0.5}
df[col] = df[col].replace(value_map)
print(col, value_map)

# 划分训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
feature_importances_ = model.feature_importances_.copy()
print('feature：importances：')
print(pd.Series(model.feature_importances_, index=df.columns.drop('y')))
print('不剔除, 决策树准确率{}'.format(acc))
model = LogisticRegression()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print('不剔除, Logistic回归准确率{}'.format(acc))
model = KNeighborsClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print('不剔除, KNN准确率{}'.format(acc))
size=10
model = MLPRegressor(hidden_layer_sizes=(size,10))
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print('不剔除, 神经网络准确率{}'.format(acc))



# 每列特征依次剔除 (此处以“housing”第五列为例)
idx = ['housing']
idx = [list(df.columns).index(i) for i in idx]

x_train_ = np.delete(x_train.copy(), idx, axis=1)
y_train_ = y_train.copy()
x_test_ = np.delete(x_test.copy(), idx, axis=1)
y_test_ = y_test.copy()
model_ = tree.DecisionTreeClassifier()
model_.fit(x_train_, y_train_)
acc_ = model_.score(x_test_, y_test_)
print('剔除第{}列, 准确率{}'.format(idx, acc_))
model_ = LogisticRegression()
model_.fit(x_train_, y_train_)
acc_ = model_.score(x_test_, y_test_)
print('剔除第{}列, logistic回归准确率{}'.format(idx, acc_))
model_ = KNeighborsClassifier()
model_.fit(x_train_, y_train_)
acc_ = model_.score(x_test_, y_test_)
print('剔除第{}列, KNN准确率{}'.format(idx, acc_))


#全部非重要特征剔除
idx = ['housing', 'loan', 'default','poutcome']
idx = [list(df.columns).index(i) for i in idx]

x_train_ = np.delete(x_train.copy(), idx, axis=1)
y_train_ = y_train.copy()
x_test_ = np.delete(x_test.copy(), idx, axis=1)
y_test_ = y_test.copy()
model_.fit(x_train_, y_train_)
acc_ = model_.score(x_test_, y_test_)
print('剔除全部, 准确率{}'.format(acc_))
model_ = LogisticRegression()
model_.fit(x_train_, y_train_)
acc_ = model_.score(x_test_, y_test_)
print('剔除全部, logistic回归准确率{}'.format( acc_))
model_ = KNeighborsClassifier()
model_.fit(x_train_, y_train_)
acc_ = model_.score(x_test_, y_test_)
print('剔除全部, KNN准确率{}'.format(acc_))


# 按照feature_importance赋权后再训练
weights = np.array([feature_importances_] * x_train.shape[0])
model1 = tree.DecisionTreeClassifier()
model1.fit(weights*x_train, y_train)
acc1 = model1.score(np.array([feature_importances_] * x_test.shape[0])*x_test, y_test)
print('赋权后, 决策树准确率{}'.format(acc1))
model1 = tree.DecisionTreeClassifier()
model1.fit(weights*x_train, y_train)
acc1 = model1.score(np.array([feature_importances_] * x_test.shape[0])*x_test, y_test)
print('赋权后, logistic回归准确率{}'.format(acc1))
model1 = KNeighborsClassifier()
model1.fit(weights*x_train, y_train)
acc1 = model1.score(np.array([feature_importances_] * x_test.shape[0])*x_test, y_test)
print('赋权后, KNN准确率{}'.format(acc1))
