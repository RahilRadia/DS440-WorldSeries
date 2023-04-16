import numpy as py
import pandas as pd
from flaml import AutoML
import tabula as tb
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#correlation heatmap
import seaborn as sns

#Auto ML Settings
automl_settings = {
    "time_budget": 60, #seconds
    "metric": 'r2',
    "task": 'regression'
    }

##load data##
#pitching_data = pd.read_csv('teampitching.csv')
#atting_data = pd.read_csv('teambattingstats.csv')
#playoffs_data = pd.read_csv('playoffappearances.csv')
train = pd.read_csv('Data/Processed/train.csv', header=0)

# Seperate predictor and response variables
X = train.iloc[:,:-1].astype(float)
Y = train["Win_Percent"]

#Split data randomly into test and train
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=12)
automl = AutoML()


#automl.fit(X_train.values, y_train, **automl_settings)
# Predict
#print(automl.predict(X_train).shape)
# Export the best model
#rint(automl.model)


#make heatmap
#plt.subplots(figsize=(15,15))
#numeric_correlations = master_ws_data.corr() # correlations between numeric variables
#sns.heatmap(numeric_correlations, xticklabels=1, yticklabels=1) 

####### Auto Feature Selection ##########

train_auto = pd.read_csv('Data/Processed/auto_feature_selection.csv', header=0)

# Seperate predictor and response variables
X_auto = train_auto.iloc[:,:-1].astype(float)
Y_auto = train_auto["Win_Percent"]

#Split data randomly into test and train
X_train_auto,X_test_auto,y_train_auto,y_test_auto=train_test_split(X_auto,Y_auto,test_size=0.3,random_state=12)
automl = AutoML()

#automl.fit(X_train_auto.values, y_train_auto, **automl_settings)

# Predict
#print(automl.predict(X_train_auto))
# Export the best model
#print(automl.model)

####### Personal Feature Selection ##########

train_per = pd.read_csv('Data/Processed/personal_feature_selection.csv', header=0)

# Seperate predictor and response variables
X_per = train_per.iloc[:,:-1].astype(float)
Y_per = train_per["Win_Percent"]

#Split data randomly into test and train
X_train_per,X_test_per,y_train_per,y_test_per=train_test_split(X_per,Y_per,test_size=0.3,random_state=12)
automl = AutoML()

automl.fit(X_train_per.values, y_train_per, **automl_settings)
print(automl.predict(X_train_auto))