import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale

#Split dataset into training et testing sets
dir_path = "./mnist-in-csv/"
train_set = pd.read_csv(dir_path + "mnist_train.csv")
train_set = np.array(train_set)
test_set = pd.read_csv(dir_path + "mnist_test.csv")
test_set = np.array(test_set)


X_train = train_set[:1000,1:]
y_train = train_set[:1000,0]

X_test = test_set[:,1:]
y_test = test_set[:,0]

#SVM training
clf = svm.SVC(kernel='rbf', gamma='scale', decision_function_shape='ovo')
clf.fit(X_train, y_train)
#Prediction
y_pred = clf.predict(X_test)

#Results
print("Accuracy score: "+ str(accuracy_score(y_test, y_pred)))
print("\nConfusion matrix: \n" + str(confusion_matrix(y_test, y_pred)))
