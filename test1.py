import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
#print(os.listdir("../input"))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
old_data = pd.read_csv('train_sample.csv')
target = old_data.is_attributed
print(old_data.columns)
def score_dataset(train_x, train_y, test_x, test_y):
    forest_model = RandomForestRegressor()
    forest_model.fit(train_x, train_y)
    predicted = forest_model.predict(test_x)
    return mean_absolute_error(test_y , predicted) 

predictors = old_data.drop(['is_attributed'], axis=1)

#this is a dataframe
data_now = predictors.select_dtypes(exclude=['object'])
print(data_now.columns)
train_x_, test_x_, train_y_, test_y_ = train_test_split(data_now,target,random_state = 0) 
val1 = score_dataset(train_x_, train_y_, test_x_, test_y_)
print(val1)