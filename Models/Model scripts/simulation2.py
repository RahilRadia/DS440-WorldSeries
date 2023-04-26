import numpy as np
import pandas as pd
class WS_sim:
    def __init__(self, sim):
        self.WSwinner = sim["WSwinner"]
        self.gameResults = sim["gameResults"]
        self.winRate = sim["winRate"]
        self.gameStats = sim["gameStats"]
        self.samples = sim["samples"]

def get_mode(v):
    uniqv, counts = np.unique(v, return_counts=True)
    return uniqv[np.argmax(counts)]

def WS_simulation(team1, year1, team2, year2, nsim):
    np.random.seed()
    mlb = mlb = pd.read_csv("Data/Final/2022_predicted_qualifiers.csv")
    
    series_results = []
    teams_playing = [f"{year1}_{team1}", f"{year2}_{team2}"]
    
    # Pulling win rate from dataset
    team1_WR = mlb.loc[(mlb['team'] == team1) & (mlb['Year'] == year1), 'Win_Percent'].values[0]
    team2_WR = mlb.loc[(mlb['team'] == team2) & (mlb['Year'] == year2), 'Win_Percent'].values[0]
    
    # List to hold the sampling results for each "game"
    game_names = ["Game1", "Game2", "Game3", "Game4", "Game5", "Game6", "Game7"]
    games = []
    
    # creating a vector of win rate as a psuedo probability
    #p = [team1_WR, team2_WR]
    
    p = [team1_WR/(team1_WR + team2_WR), team2_WR/(team1_WR + team2_WR)]
    sampling_results = np.random.choice(teams_playing, size=(nsim, 7), replace=True, p=p)
    
    for i in range(7):
        # Putting most sampled team into results vector
        series_results.append(get_mode(sampling_results[:, i]))
        
        # Putting sampling results into previously made lists
        games.append(pd.DataFrame({
            'Year': [year1, year2],
            'teams': [team1, team2],
            'nSampled': [np.count_nonzero(sampling_results[:, i] == f"{year1}_{team1}"),
                         np.count_nonzero(sampling_results[:, i] == f"{year2}_{team2}")],
            'perSampled': [np.count_nonzero(sampling_results[:, i] == f"{year1}_{team1}") / nsim,
                           np.count_nonzero(sampling_results[:, i] == f"{year2}_{team2}") / nsim],
            'winRate': p
        }))
        
    games = dict(zip(game_names, games))
        
    sim = {
        'WSwinner': get_mode(series_results),
        'gameResults': series_results,
        'winRate': p,
        'gameStats': games,
        'samples': sampling_results,
    }
    
    return sim

cle_tor = WS_simulation("CLE", 2022, "TOR", 2022, 50)
print(cle_tor)