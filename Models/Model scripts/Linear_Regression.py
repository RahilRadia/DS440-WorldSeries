import numpy as py
import pandas as pd
import tabula as tb
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


##load data##
#pitching_data = pd.read_csv('teampitching.csv')
#atting_data = pd.read_csv('teambattingstats.csv')
#playoffs_data = pd.read_csv('playoffappearances.csv')
master_ws_data = pd.read_csv('Data/Processed/master_ws_data.csv', header=0)


lr = LinearRegression()
# Seperate predictor and response variables
X = master_ws_data.iloc[:,:-1].astype(float)
Y = master_ws_data["Win_Percent"]

print(X)
print(Y)

#Split data randomly into test and train
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=12)

#Fit linear regression using training data and test for accuracy using train
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2) #R^2 Metric for accuracy



