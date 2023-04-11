from featurewiz import featurewiz
import pandas as pd

train = pd.read_csv('Data/Processed/train.csv', header=0)

#Set Response Variable
y = 'Win_Percent'
#Featurewiz feature selection
#Use SULOV algorithm to find variables with high correlation to response and low correlation to other predictors
features, auto_fs = featurewiz(train, target = y, corr_limit = 0.5, verbose = 2, sep = ",", header = 0, test_data="", feature_engg="", category_encoders="")

pandas_df = auto_fs.collect()
auto_fs = pd.DataFrame(pandas_df, columns=auto_fs.columns)

train.to_csv('Data/Processed/auto_feature_selection.csv')
print(features)

#Personal Feature Selection
personal_fs = train.columns["ERA", "ERAgainst", "AVGbat", "OPS", "WPApitch", "Batting", "HR", "ER", "WARpitch", "FIP", "WHIPAllowed", "RBI", "KtoBB", "WARbat"]
print(personal_fs)