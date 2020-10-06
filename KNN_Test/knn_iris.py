from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris=datasets.load_iris()
x=iris.data
y=iris.target
print(iris.keys())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
acc=knn.score(x_test,y_test)
print('分类准确率为：%f'%acc)