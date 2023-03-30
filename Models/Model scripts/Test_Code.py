import numpy as py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from flaml import AutoML
import tabula as tb
import re
from sklearn.model_selection import train_test_split



##load data##
pitching_data = pd.read_csv('teampitching.csv')
batting_data = pd.read_csv('teambattingstats.csv')
playoffs_data = pd.read_csv('playoffappearances.csv')
master_ws_data = pd.read_csv('masterWS.csv', header=0)

###CLEAN DATA###

#create predictor variable Win_Percent as percent wins in the season
master_ws_data["Win_Percent"] = master_ws_data["W"] / (master_ws_data["W"] + master_ws_data["L"])
#Remove the Wins/Loses columns and case identifier variables Season and Team
master_ws_data = master_ws_data.drop(columns = ["Season","Team",'W', 'L'])
#Cast strings to floats and remove JSON char
master_ws_data= master_ws_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
master_ws_data = master_ws_data.replace('%', '', regex=True)
master_ws_data = master_ws_data.astype(float)
print(master_ws_data.head())

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

#correlation heatmap
import seaborn as sns

plt.subplots(figsize=(15,15))
numeric_correlations = master_ws_data.corr() # correlations between numeric variables
sns.heatmap(numeric_correlations, xticklabels=1, yticklabels=1)





