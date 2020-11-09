import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 加载数据
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
# 使用线性回归算法
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size=0.3,random_state=5)
lr = LinearRegression().fit(X_train, y_train)#拟合模型

print("lr.coef_: {}".format(lr.coef_))  # 输出系数（斜率）
print("lr.intercept_: {}".format(lr.intercept_))  # 输出截距（偏移量）

# 准确率（模型效果）
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#用训练好的模型求新数据的值
predictResult = lr.predict(X_test)
print(predictResult)

