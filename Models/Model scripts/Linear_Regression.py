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

#import datasets
original = pd.read_csv('Data/Processed/train.csv')
auto = pd.read_csv('Data/Processed/train_auto.csv', header=0)
personal = pd.read_csv('Data/Processed/train_per.csv', header=0)

lr = LinearRegression()

######## Original Data #######
X = original.iloc[:,:-1].astype(float)
Y = original["Win_Percent"]

print(X)
print(Y)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=12)

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Original Dataset R2 score:", r2) #R^2 Metric for accuracy


######### Auto FS ##########
X_auto = auto.iloc[:,:-1].astype(float)
Y_auto = auto["Win_Percent"]

print(X_auto)
print(Y_auto)

X_train_auto,X_test_auto,y_train_auto,y_test_auto = train_test_split(X_auto,Y_auto,test_size=0.3,random_state=12)

lr.fit(X_train_auto,y_train_auto)
y_pred_auto = lr.predict(X_test_auto)
r2 = r2_score(y_test_auto, y_pred_auto)
print("Auto feature selection R2 score:", r2)


######## Personal FS #########
X_per = personal.iloc[:,:-1].astype(float)
Y_per = personal["Win_Percent"]

print(X_per)
print(Y_per)

X_train_per,X_test_per,y_train_per,y_test_per = train_test_split(X_per,Y_per,test_size=0.3,random_state=12)

lr.fit(X_train_per,y_train_per)
y_pred_per = lr.predict(X_test_per)
r2 = r2_score(y_test_per, y_pred_per)
print("Personal feature selection R2 score:", r2) 



