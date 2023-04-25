import pandas as pd
import numpy as np

def getmode(v):
    uniqv = np.unique(v)
    return uniqv[np.argmax(np.bincount(np.where(v == uniqv[:,None])[0]))]



# The probability of success will be determined by the win/loss rate the team had during the regular season for that particular year.
def WS_simulation(team1,year1, team2, year2, nsim):
    np.random.seed(int(pd.Timestamp.now().timestamp())) # set seed
    mlb = pd.read_csv("Data/Final/2022_predicted_qualifiers.csv") #this changes to our data
    seriesResults = [] # create empty list to hold results
    
    teamsPlaying = [f"{year1}_{team1}", f"{year2}_{team2}"]
    
    # Pulling win rate from dataset
    team1WR = mlb.loc[(mlb['team'] == team1) & (mlb['Year'] == year1), 'Win_Percent'].values[0]
    team2WR = mlb.loc[(mlb['team'] == team2) & (mlb['Year'] == year2), 'Win_Percent'].values[0]

    # List to hold the sampling results for each "game"
    gameNames = ["Game1", "Game2", "Game3", "Game4", "Game5", "Game6", "Game7"] 
    games = {}
 
    # creating a vector of win rate as a psuedo probability
    p = [team1WR, team2WR]
   
    samplingResults = np.random.choice(teamsPlaying, size=(nsim, 7), replace=True, p=p).tolist()

    for i in range(7):
        # Putting most sampled team into results vector
        seriesResults.append(getmode(samplingResults[:, i]))

        # Putting sampling results into previously made lists
        games[gameNames[i]] = pd.DataFrame({
            'Year': [year1, year2],
            'teams': [team1, team2],
            'nSampled': [samplingResults[:, i].count(f"{year1}_{team1}"), samplingResults[:, i].count(f"{year2}_{team2}")],
            'perSampled': [samplingResults[:, i].count(f"{year1}_{team1}") / nsim, samplingResults[:, i].count(f"{year2}_{team2}") / nsim],
            'winRate': p
        })

    sim = {
        'WSwinner': getmode(seriesResults),
        'gameResults': seriesResults,
        'winRate': p,
        'gameStats': games,
        'samples': samplingResults
    }
hou_atl = WS_simulation('HOU','2022', 'ATL','2022', 3)
print(hou_atl)