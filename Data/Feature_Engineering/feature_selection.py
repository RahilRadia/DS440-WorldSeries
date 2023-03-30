from featurewiz import featurewiz
import pandas as pd

master_ws_data = pd.read_csv('Data/Processed/master_ws_data.csv', header=0)

y = 'Win_Percent'

features, train = featurewiz(master_ws_data, target = y, corr_limit = 0.7, verbose = 2, sep = ",", header = 0, test_data="", feature_engg="", category_encoders="")



