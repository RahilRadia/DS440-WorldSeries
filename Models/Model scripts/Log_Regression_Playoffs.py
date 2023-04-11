# as shown in previous section
import sklearn

dimensions = ["ops", "r_per_game", "ra_per_game", "make_playoffs"]
predictor_dimensions = ["ops", "r_per_game", "ra_per_game"] # choose anything for these. OPS, R/G, RA/G, etc. 
_df = df[dimensions]


#std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
#std_clf.fit(X_train, y_train)

# predict playoff qualifiers for a given year (2000-2015)
#for year in range(2015, 2021):
#    _df = df[df["year"]==year]
#    X = _df[predictor_dimensions]
#    y = _df["make_playoffs"]

#    _df["predicted_playoff_qualifier"] = std_clf.predict(X)
#    print("--------")
#    print(year)
#    print("--------")
#    print("Predicted playoff qualifiers in " + str(year) + ":")
#    predicted = set(_df[_df["predicted_playoff_qualifier"]==True]["franchise_id"])
#    print(predicted)
#    print()
#    print()


# franchise id means the 3 letter abbreviation for team
# idk what predictor dimensions should be runs per game and runs allowed per game sound good
# predict 2022 based on who wins from 2015-2021

