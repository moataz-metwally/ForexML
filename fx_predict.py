#!/usr/bin/python
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import  metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
df1 = pd.read_csv("USDJPY.csv", names=["time", "open", "high", "low", "close", "volume"], header=None)
h = .02  # step size in the mesh
open_data = pd.to_numeric(df1['open'][1:])
close_data = pd.to_numeric(df1['close'][1:])
low_data = pd.to_numeric(df1['low'][1:])
high_data = pd.to_numeric(df1['high'][1:])
volume_data = pd.to_numeric(df1['volume'][1:])
close_relative = ((close_data - open_data) / open_data) * 10000
high_relative = ((high_data - open_data) / open_data) * 10000
low_relative = ((low_data - open_data) / open_data) * 10000

# Creat series for the output data
predict_data = pd.Series()

# Creat data frame for the input data
input_data = pd.DataFrame(dtype=float)



#filling predict data
for i in range(11, open_data.size):
        if (open_data.values[i] > close_data.values[i]):
            predict_data=predict_data.append(pd.Series(data=[0]),ignore_index=True)
        else:
            predict_data=predict_data.append(pd.Series(data=[1]),ignore_index=True)

print(predict_data.shape)

#filling input data
for i in range(1, open_data.size-10):
    ar = pd.Series()
    for n in range(i,i+10):
        ar= ar.append(pd.Series(data=[close_relative.values[n],high_relative.values[n],low_relative.values[n],volume_data.values[n]]), ignore_index=True,verify_integrity=True)

    input_data=input_data.append(pd.Series(data=ar),ignore_index=True,verify_integrity=True)

#input_data.to_csv(path_or_buf="mm.csv")

clf = linear_model.LogisticRegression(C=1e10,max_iter=100000000, verbose=True)
#clf = SVC(kernel='poly',max_iter=1000000)
#clf = MLPClassifier(solver='sgd', alpha=1e-5,
 #                    hidden_layer_sizes=(50, 50), random_state=1,max_iter=600000)



print(input_data.values.shape)
X = input_data.values
Y = predict_data.values

# get test data and train data
x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=5)

# we create an instance of Neighbours Classifier and fit the data.
#logreg.fit(x_train, x_train)
print(y_train.shape)
clf.fit(x_train, y_train)

y_predict = clf.predict(x_train)
#print out the accuracy of the model
print("=============")
print(pd.Series(y_predict).value_counts())
print("=============")
print(pd.Series(y_train).value_counts())
print("=============")
print(metrics.accuracy_score(y_train,y_predict))
print("=============")
t= metrics.confusion_matrix(y_train,y_predict)
print(t[0,0]) # true negative
print(t[0,1]) # true postive
print(t[1,0]) # false negative
print(t[1,1]) # t