from msilib import Feature
from featurewiz import featurewiz
import pandas as pd

train = pd.read_csv('Data/Processed/train.csv', header=0)

#Set Response Variable
y = 'Win_Percent'

#Featurewiz feature selection
#Use SULOV algorithm to find variables with high correlation to response and low correlation to other predictors
features, auto_fs = featurewiz(train, target = y, corr_limit = 0.5, verbose = 2, sep = ",", header = 0, test_data="", feature_engg="", category_encoders="")

#selected_features = pd.DataFrame(auto_fs, columns=['Feature'])
auto_fs.to_csv('Data/Processed/train_auto.csv')

#Personal Feature Selection
personal_fs = train[["ERA", "ERAgainst", "AVGbat", "OPS", "WPApitch", "Batting", "HR", "EV", "WARpitch", "FIP", "WHIPAllowed", "RBI", "KtoBB", "WARbat", "Win_Percent"]]
personal_fs.to_csv('Data/Processed/train_per.csv')


###Post Testing###
test = pd.read_csv('Data/Processed/test.csv', header=0)
test_auto = test[features]
test_per = test[["ERA", "ERAgainst", "AVGbat", "OPS", "WPApitch", "Batting", "HR", "EV", "WARpitch", "FIP", "WHIPAllowed", "RBI", "KtoBB", "WARbat"]]

test_auto.to_csv('Data/Processed/test_auto.csv')
test_per.to_csv('Data/Processed/test_per.csv')


