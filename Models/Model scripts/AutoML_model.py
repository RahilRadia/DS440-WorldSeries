import numpy as py
import pandas as pd
from flaml import AutoML, DataTransformer
import tabula as tb
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Auto ML Settings
automl_settings = {
    "time_budget": 90,
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

train_team = pd.read_csv('Data/Processed/train_team.csv', header = 0)
test_team = pd.read_csv('Data/Processed/test_team.csv', header = 0)

################### Original Data ########################
#Seperate predictor and response variables
#X_train = train.iloc[:,:-1]
#Y_train = train["Win_Percent"]

full_train = pd.merge(train_team, train, left_index=True, right_index=True)
full_test = pd.merge(test_team, test, left_index=True, right_index=True)


X_train = full_train.iloc[:,:-1]
Y_train = full_train["Win_Percent"]
X_test = full_test

# Specify the categorical variables
cat_cols = ['season', 'team', 'league']

# Initialize the DataTransformer
transformer = DataTransformer(cat_cols=cat_cols)

# Fit the DataTransformer on the training data
transformer.fit(X_train)

# Transform the training data
X_train_transformed = transformer.transform(X_train)

# Fit the AutoML model on the transformed data
automl = AutoML()
automl.fit(X_train_transformed, Y_train, **automl_settings)

#automl = AutoML()
#automl.fit(X_train.values, Y_train, **automl_settings)

#predict
Y_test = automl.predict(X_test)
Y_test.head()

############### Auto Feature Selection Data ##################
# Seperate predictor and response variables
X_train_auto = train_auto.iloc[:,:-1]
Y_train_auto = train_auto["Win_Percent"]

X_test_auto = test_auto

automl_auto = AutoML()
automl_auto.fit(X_train_auto.values, Y_train_auto, **automl_settings)


################### Personal Feature Selection Data ##################
# Seperate predictor and response variables
X_train_per = train_per.iloc[:,:-1]
Y_train_per = train_per["Win_Percent"]

X_test_per = test_per


automl_per = AutoML()
automl_per.fit(X_train_per.values, Y_train_per, **automl_settings)



########################################################## PART 2 #########################################################################
#Test dataset






