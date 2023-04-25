import random
import numpy as np
import pandas as pd
import time

class WS_sim:
    def __init__(self, WSwinner, gameResults, winRate, gameStats, samples):
        self.WSwinner = WSwinner
        self.gameResults = gameResults
        self.winRate = winRate
        self.gameStats = gameStats
        self.samples = samples

def getmode(v):
    uniqv = list(set(v))
    return uniqv[max(range(len([v.count(i) for i in uniqv])), key=[v.count(i) for i in uniqv].__getitem__)]

def WS_simulation(team1, year1, team2, year2, nsim):
    np.random.seed(int(time.time())) 
    mlb = pd.read("Data/Final/2022_predicted_qualifiers") #this changes to our data
    
    seriesResults = []
    teamsPlaying = [str(year1) + "_" + team1, str(year2) + "_" + team2]
    
    # Pulling win rate from dataset
    team1WR = mlb.loc[(mlb['team'] == team1) & (mlb['Year'] == year1), 'WR'].values[0]
    team2WR = mlb.loc[(mlb['team'] == team2) & (mlb['Year'] == year2), 'WR'].values[0]
    
    # List to hold the sampling results for each "game"
    gameNames = ["Game1", "Game2", "Game3", "Game4", "Game5", "Game6", "Game7"] 
    games = dict()
    
    # creating a vector of win rate as a psuedo probability
    p = [team1WR, team2WR]
    
    samplingResults = np.zeros((nsim, 7), dtype='str')
    
    for i in range(7):
        # Sampling teams with replacement using win rate as probabilities
        samplingResults[:, i] = np.random.choice(teamsPlaying, size=nsim, replace=True, p=p)
        
        # Putting most sampled team into results vector
        seriesResults.append(getmode(samplingResults[:, i]))
        
        # Putting sampling results into previously made lists
        games[gameNames[i]] = pd.DataFrame({
            'Year': [year1, year2],
            'teams': [team1, team2],
            'nSampled': np.bincount(np.where(samplingResults[:, i] == teamsPlaying[0])[0], minlength=nsim).tolist(),
            'perSampled': np.bincount(np.where(samplingResults[:, i] == teamsPlaying[0])[0], minlength=nsim).tolist() / nsim + np.bincount(np.where(samplingResults[:, i] == teamsPlaying[1])[0], minlength=nsim).tolist() / nsim,
            'winRate': p
        })
    
    sim = WS_sim(
        WSwinner=getmode(seriesResults),
        gameResults=seriesResults,
        winRate=p,
        gameStats=games,
        samples=samplingResults.tolist()
    )
    
    return sim
#example   
hou21_atl21 = WS_simulation('HOU', 2021, 'ATL', 2021, 5)
print(hou21_atl21)
