from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# 加载数据
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
# 使用随机森林算法
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size=0.3,random_state=5)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
predict_results=clf.predict(X_test)
print('分类准确率分数:')
print(accuracy_score(predict_results, y_test))
conf_mat = confusion_matrix(y_test, predict_results)
print('混淆矩阵:')
print(conf_mat)