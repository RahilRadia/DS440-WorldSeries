import numpy as py
import pandas as pd
import re

##load data##
playoffs_data = pd.read_csv('Data/Raw/playoffappearances.csv')
master_ws_data = pd.read_csv('Data/Raw/masterWorldSeries.csv', header=0)
data_2023 = pd.read_csv('Data/Raw/2023STATS.csv', header = 0)



###CLEAN DATA###

#Split into 2015-2021 and 2022 seasons for test and train
master_test = master_ws_data.loc[master_ws_data['season'] == 2022]
master_train = master_ws_data.loc[master_ws_data['season'] != 2022]#create predictor variable Win_Percent as percent wins in the season
master_train["Win_Percent"] = master_ws_data["w"] / (master_ws_data["w"] + master_ws_data["l"])

#Add season and team name to array
test_team = master_test.iloc[:,1:3]
train_team = master_train.iloc[:,1:3]
teams_23 = data_2023[['team']]

print(teams_23.head())

#Remove the Wins/Loses columns and case identifier variables Season and Team
master_test = master_test.drop(columns = ["season","team", "league", 'w', 'l'])
master_train = master_train.drop(columns = ["season","team", "league", 'w', 'l'])
data_2023 = data_2023.drop(columns = ["team"])



#Cast strings to floats and remove JSON char
master_train = master_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
master_train = master_train.replace('%', '', regex=True)
master_train = master_train.astype(float)

master_test= master_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
master_test = master_test.replace('%', '', regex=True)
master_test = master_test.astype(float)

data_2023= data_2023.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
data_2023 = data_2023.replace('%', '', regex=True)
data_2023 = data_2023.astype(float)


#Save to processed data folder
master_train.to_csv('Data/Processed/train.csv')
master_test.to_csv('Data/Processed/test.csv')

train_team.to_csv('Data/Processed/train_team.csv')
test_team.to_csv('Data/Processed/test_team.csv')

data_2023.to_csv('Data/Processed/test_23.csv')
teams_23.to_csv('Data/Processed/team_23.csv')

