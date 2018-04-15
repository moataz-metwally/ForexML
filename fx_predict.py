#!/usr/bin/python
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import  metrics
df1 = pd.read_csv("USDJPY.csv", names=["time", "open", "high", "low", "close", "volume"], header=None)
h = .02  # step size in the mesh
open_data = pd.to_numeric(df1['open'][1:])
close_data = pd.to_numeric(df1['close'][1:])
low_data = pd.to_numeric(df1['low'][1:])
high_data = pd.to_numeric(df1['high'][1:])
volume_data = pd.to_numeric(df1['volume'][1:])
print(type(open_data[1]))
print(open_data[open_data.size])
close_relative = ((close_data - open_data) / open_data) * 10000
high_relative = ((high_data - open_data) / open_data) * 10000
low_relative = ((low_data - open_data) / open_data) * 10000

# Creat series for the output data
predict_data = pd.Series()

# Creat data frame for the input data
input_data = pd.DataFrame(dtype=float,columns=['bar1_relative_close',
                                                        'bar1_relative_low',
                                                        'bar1_relative_high',
                                                        'bar1_volume',
                                                        
                                                        'bar2_relative_close',
                                                        'bar2_relative_low',
                                                        'bar2_relative_high',
                                                        'bar2_volume',
                                                        
                                                        'bar3_relative_close',
                                                        'bar3_relative_low',
                                                        'bar3_relative_high',
                                                        'bar3_volume',
                                                        
                                                        'bar4_relative_close',
                                                        'bar4_relative_low',
                                                        'bar4_relative_high',
                                                        'bar4_volume'])
print(type(open_data.values[0]))
print(close_relative)


#filling predict data
for i in range(5, open_data.size-3):
        if (open_data.values[i] > close_data.values[i+3]):
            predict_data=predict_data.append(pd.Series(data=[0]))
        else:
            predict_data=predict_data.append(pd.Series(data=[1]))

print(predict_data.shape)

#filling input data
for i in range(1, open_data.size-7):
    ar = pd.Series(data=[close_relative.values[i],
                         low_relative.values[i],
                         high_relative.values[i],
                         volume_data.values[i],

                         close_relative.values[i+1],
                         low_relative.values[i+1],
                         high_relative.values[i+1],
                         volume_data.values[i+1],

                         close_relative.values[i+2],
                         low_relative.values[i+2],
                         high_relative.values[i+2],
                         volume_data.values[i+2],

                         close_relative.values[i+3],
                         low_relative.values[i+3],
                         high_relative.values[i+3],
                         volume_data.values[i+3]
                         ],index=['bar1_relative_close',
                                                        'bar1_relative_low',
                                                        'bar1_relative_high',
                                                        'bar1_volume',
                                                        
                                                        'bar2_relative_close',
                                                        'bar2_relative_low',
                                                        'bar2_relative_high',
                                                        'bar2_volume',
                                                        
                                                        'bar3_relative_close',
                                                        'bar3_relative_low',
                                                        'bar3_relative_high',
                                                        'bar3_volume',
                                                        
                                                        'bar4_relative_close',
                                                        'bar4_relative_low',
                                                        'bar4_relative_high',
                                                        'bar4_volume'] )

    input_data=input_data.append(pd.Series(data=ar),ignore_index=True)

#input_data.to_csv(path_or_buf="mm.csv")

logreg = linear_model.LogisticRegression(C=1e5)

print(input_data.values.shape)
X = input_data.values
Y = predict_data.values

# get test data and train data
x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=0)
print(Y.shape)
print(x_test)
# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(x_train, y_train)

y_predict = logreg.predict(x_test)
#print out the accuracy of the model
print(metrics.accuracy_score(y_test,y_predict))