import seaborn as sns

# calculate correlation matrix
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(50,50))

# plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig('/content/drive/My Drive/DS440/corr_matrix.png')

#Correlation with output variable
cor_target = abs(corr_matrix["Win_Percent"])
#Selecting highly correlated features
relevant_features_percent = cor_target[cor_target>0.684]
relevant_features_percent