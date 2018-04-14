import pandas as pd

df1 = pd.read_csv("~/Downloads/USDJPY.csv", names=["time", "open", "high", "low", "close", "volume"], header=None)

open_data = df1['open'][1:].convert_objects(convert_numeric=True)
close_data = df1['close'][1:].convert_objects(convert_numeric=True)
low_data = df1['low'][1:].convert_objects(convert_numeric=True)
high_data = df1['high'][1:].convert_objects(convert_numeric=True)
volume_data = df1['volume'][1:].convert_objects(convert_numeric=True)
print(type(open_data))
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

#filling predict data
for i in range(5, open_data.size):
    if (open_data.size > (i + 3)):
        if (open_data.values[i] > close_data.values[i+3]):
            predict_data=predict_data.append(pd.Series(data=[0],index=[i]))
        else:
            predict_data=predict_data.append(pd.Series(data=[1],index=[i]))

print(predict_data.shape)

#filling input data
for i in range(1, open_data.size-4):
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


print(input_data)
