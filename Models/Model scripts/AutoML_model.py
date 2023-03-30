import numpy as py
import pandas as pd
from flaml import AutoML
import tabula as tb
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#correlation heatmap
import seaborn as sns

##load data##
#pitching_data = pd.read_csv('teampitching.csv')
#atting_data = pd.read_csv('teambattingstats.csv')
#playoffs_data = pd.read_csv('playoffappearances.csv')
master_ws_data = pd.read_csv('Data/Processed/master_ws_data.csv', header=0)

# Seperate predictor and response variables
X = master_ws_data.iloc[:,:-1].astype(float)
Y = master_ws_data["Win_Percent"]

print(X)
print(Y)

#Split data randomly into test and train
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=12)
automl = AutoML()

#Auto ML Settings
automl_settings = {
    "time_budget": 60, #seconds
    "metric": 'r2',
    "task": 'regression'
    }


#automl.fit(X_train.values, y_train, **automl_setting,)
# Predict
print(automl.predict(X_train).shape)
# Export the best model
print(automl.model)


#make heatmap
plt.subplots(figsize=(15,15))
numeric_correlations = master_ws_data.corr() # correlations between numeric variables
sns.heatmap(numeric_correlations, xticklabels=1, yticklabels=1) 