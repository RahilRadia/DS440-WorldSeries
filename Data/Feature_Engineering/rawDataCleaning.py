import numpy as py
import pandas as pd
import re

##load data##
pitching_data = pd.read_csv('Data/Raw/teampitching.csv')
batting_data = pd.read_csv('Data/Raw/teambattingstats.csv')
playoffs_data = pd.read_csv('Data/Raw/playoffappearances.csv')
master_ws_data = pd.read_csv('Data/Raw/masterWS.csv', header=0)

###CLEAN DATA###

#create predictor variable Win_Percent as percent wins in the season
master_ws_data["Win_Percent"] = master_ws_data["W"] / (master_ws_data["W"] + master_ws_data["L"])


#Split into 2015-2021 and 2022 seasons for test and train
master_test = master_ws_data.loc[master_ws_data['Season'] == 2022]

#Remove the Wins/Loses columns and case identifier variables Season and Team
master_test = master_ws_data.drop(columns = ["Season","Team",'W', 'L'])
master_test = master_ws_data.drop(columns = ["Season","Team",'W', 'L'])


#Cast strings to floats and remove JSON char
master_ws_data= master_ws_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
master_ws_data = master_ws_data.replace('%', '', regex=True)
master_ws_data = master_ws_data.astype(float)

#Save to processed data folder
master_ws_data.to_csv('Data/Processed/master_ws_data.csv')
master_test.to_csv('Data/Processed/master_test.csv')
