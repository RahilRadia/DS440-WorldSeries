import numpy as py
import pandas as pd
from flaml import AutoML
import tabula as tb
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


train_auto = pd.read_csv('Data/Processed/train_auto.csv', header = 0)
test_auto = pd.read_csv('Data/Processed/test_auto.csv', header = 0)
test_23 = pd.read_csv('Data/Processed/test_23.csv', header = 0)

train_team = pd.read_csv('Data/Processed/train_team.csv', header = 0)
test_team = pd.read_csv('Data/Processed/test_team.csv', header = 0)




#full_train = pd.merge(train_team, train_auto, left_index=True, right_index=True)
#full_test = pd.merge(test_team, test_auto, left_index=True, right_index=True)

X_train_auto = train_auto.iloc[:,:-1]
Y_train_auto = train_auto["Win_Percent"]
X_test_auto = test_auto



#Test dataset
#auto fs
with open('flaml_model_auto.pkl', 'rb') as f:
    flaml_model_auto = pickle.load(f)

#predict
Y_test_auto = flaml_model_auto.predict(X_test_auto)
Y_test_23 = flaml_model_auto.predict(test_23)

print(Y_test_23)
print(Y_test_auto)

predictions = test_team
predictions['Win_Percent'] = Y_test_auto.tolist()

predictions_2023 = test_team
predictions_2023['Win_Percent'] = Y_test_23.toList()


playoff_qualifiers = predictions.groupby('league', group_keys=True).apply(lambda x: x.nlargest(6, ['Win_Percent']))
playoff_qualifiers_2023 = predictions_2023.groupby('league', group_keys=True).apply(lambda x: x.nlargest(6, ['Win_Percent']))


playoff_qualifiers_2023.to_csv("Data/Final/2023_predicted_qualifiers.csv")
