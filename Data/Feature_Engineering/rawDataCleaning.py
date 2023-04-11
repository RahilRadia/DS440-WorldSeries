import numpy as py
import pandas as pd
import re

##load data##
playoffs_data = pd.read_csv('Data/Raw/playoffappearances.csv')
master_ws_data = pd.read_csv('Data/Raw/masterWorldSeries.csv', header=0)

###CLEAN DATA###

#Split into 2015-2021 and 2022 seasons for test and train
master_test = master_ws_data.loc[master_ws_data['season'] == 2022]
master_train = master_ws_data.loc[master_ws_data['season'] != 2022]#create predictor variable Win_Percent as percent wins in the season
master_train["Win_Percent"] = master_ws_data["w"] / (master_ws_data["w"] + master_ws_data["l"])

#Remove the Wins/Loses columns and case identifier variables Season and Team
master_test = master_test.drop(columns = ["season","team",'w', 'l'])
master_train = master_train.drop(columns = ["season","team",'w', 'l'])

#Cast strings to floats and remove JSON char
master_train = master_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
master_train = master_train.replace('%', '', regex=True)
master_train = master_train.astype(float)

master_test= master_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
master_test = master_test.replace('%', '', regex=True)
master_test = master_test.astype(float)

#Save to processed data folder
master_train.to_csv('Data/Processed/train.csv')
master_test.to_csv('Data/Processed/test.csv')
