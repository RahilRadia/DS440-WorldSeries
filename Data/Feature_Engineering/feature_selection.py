from featurewiz import featurewiz
import pandas as pd

train = pd.read_csv('Data/Processed/train.csv', header=0)

#Set Response Variable
y = 'Win_Percent'

#Featurewiz feature selection
#Use SULOV algorithm to find variables with high correlation to response and low correlation to other predictors
features, auto_fs = featurewiz(train, target = y, corr_limit = 0.5, verbose = 2, sep = ",", header = 0, test_data="", feature_engg="", category_encoders="")
print(features)
print(auto_fs)


#selected_features = pd.DataFrame(auto_fs, columns=['Feature'])
auto_fs.to_csv('Data/Processed/auto_feature_selection.csv')
print(features)

#Personal Feature Selection
personal_fs = train[["ERA", "ERAgainst", "AVGbat", "OPS", "WPApitch", "Batting", "HR", "EV", "WARpitch", "FIP", "WHIPAllowed", "RBI", "KtoBB", "WARbat", "Win_Percent"]]
personal_fs.to_csv('Data/Processed/personal_feature_selection.csv')


###Post Testing###