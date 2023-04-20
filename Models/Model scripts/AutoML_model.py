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
    "time_budget": 90, #seconds
    "metric": 'r2',
    "task": 'regression'
    }

####load data#####
train = pd.read_csv('Data/Processed/train.csv', header=0)
test = pd.read_csv('Data/Processed/test.csv', header=0)

train_auto = pd.read_csv('Data/Processed/train_auto.csv', header = 0)
test_auto = pd.read_csv('Data/Processed/test_auto.csv', header = 0)

train_per = pd.read_csv('Data/Processed/train_per.csv', header = 0)
test_per = pd.read_csv('Data/Processed/test_per.csv', header = 0)



################### Original Data ########################
#Seperate predictor and response variables
X = train.iloc[:,:-1].astype(float)
Y = train["Win_Percent"]

X_test = test.iloc[:].astype(float)

automl = AutoML()
automl.fit(X.values, Y, **automl_settings)

#predict
print(automl.predict().shape)
# Export the best model



############### Auto Feature Selection Data ##################
# Seperate predictor and response variables
X_auto = train_auto.iloc[:,:-1].astype(float)
Y_auto = train_auto["Win_Percent"]

X_test_auto = test_auto.iloc[:].astype(float)

automl_auto = AutoML()
automl.fit(X_train_auto.values, y_train_auto, **automl_settings)

# Predict
#print(automl.predict(X_test_auto))
# Export the best model
#print(automl.model)

################### Personal Feature Selection Data ##################

train_per = pd.read_csv('Data/Processed/personal_feature_selection.csv', header=0)

# Seperate predictor and response variables
X_per = train_per.iloc[:,:-1].astype(float)
Y_per = train_per["Win_Percent"]

#Split data randomly into test and train
#X_train_per,X_test_per,y_train_per,y_test_per=train_test_split(X_per,Y_per,test_size=0.3,random_state=12)

automl = AutoML()

print(X_test_per)


automl.fit(X_train_per.values, y_train_per, **automl_settings)

yfit = automl.predict(X_test_per)
print(yfit)


########################################################## PART 2 #########################################################################
#Test dataset






