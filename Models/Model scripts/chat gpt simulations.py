import random
import numpy as np
import pandas as pd
import time
from collections import Counter
from scipy.stats import mode
import warnings

# Ignore future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class WS_sim:
    def __init__(self, WSwinner, gameResults, winRate, gameStats, samples):
        self.WSwinner = WSwinner
        self.gameResults = gameResults
        self.winRate = winRate
        self.gameStats = gameStats
        self.samples = samples


def getmode(v):
    values, counts = np.unique(v, return_counts=True)
    max_count = np.max(counts)
    if np.sum(counts == max_count) > 1:
        return random.choice(modes)  # there are multiple modes
    else:
        return values[np.argmax(counts)]
    #uniqv = list(set(v))
    #return uniqv[max(range(len([v.count(i) for i in uniqv])), key=[v.count(i) for i in uniqv].__getitem__)]

def WS_simulation(team1, year1, team2, year2, nsim):
    np.random.seed(int(time.time())) 
    mlb = pd.read_csv("Data/Final/2022_predicted_qualifiers.csv") #this changes to our data
    
    seriesResults = []
    teamsPlaying = [team1, team2]
    
    # Pulling win rate from dataset
    team1WR = mlb.loc[(mlb['team'] == team1) & (mlb['Year'] == year1), 'Win_Percent'].values[0]
    team2WR = mlb.loc[(mlb['team'] == team2) & (mlb['Year'] == year2), 'Win_Percent'].values[0]
    
    

    # List to hold the sampling results for each "game"
    gameNames = ["Game1", "Game2", "Game3", "Game4", "Game5", "Game6", "Game7"] 
    games = dict()
    
    # creating a vector of win rate as a psuedo probability
    p = [team1WR/(team1WR + team2WR), team2WR/(team1WR + team2WR)]
    
    samplingResults = np.zeros((nsim, 7), dtype='str')
    
    


    for i in range(7):
    # Sampling teams with replacement using win rate as probabilities
        samplingResults = np.random.choice(teamsPlaying, size=nsim, replace=True, p=p)

        mode, count = pd.Series(samplingResults).mode().values, pd.Series(samplingResults).value_counts().values
        if len(mode) > 1:
            # If there is no unique mode, randomly choose a winner among the teams that were most frequently sampled
            winner = np.random.choice(mode, size=1)[0]
        else:
            winner = mode[0]
    # Putting most sampled team into results vector



        seriesResults.append(winner)
        
    # Putting sampling results into previously made lists
        if gameNames[i] not in games:
            games[gameNames[i]] = pd.DataFrame()
        
        games[gameNames[i]] = games[gameNames[i]].append({
            'Year': [year1, year2],
            'teams': [team1, team2],
            'nSampled': np.bincount(np.where(samplingResults == teamsPlaying[0])[0], minlength=nsim).tolist(),
            'perSampled': (np.bincount(np.where(samplingResults == teamsPlaying[0])[0], minlength=nsim) + np.bincount(np.where(samplingResults == teamsPlaying[1])[0], minlength=nsim) / nsim).tolist(),
            'winRate': p
        }, ignore_index=True)


    #for i in range(7):
    #    # Sampling teams with replacement using win rate as probabilities
    #    samplingResults = np.random.choice(teamsPlaying, size=nsim, replace=True, p=p)
    #    # Putting most sampled team into results vector
    #    seriesResults.append(getmode(samplingResults))
        
    #    # Putting sampling results into previously made lists
    #    games[gameNames[i]] = pd.DataFrame({
    #        'Year': [year1, year2],
    #        'teams': [team1, team2],
    #        'nSampled': [1,2],#np.bincount(np.where(samplingResults == teamsPlaying[0])[0], minlength=nsim).tolist(),
    #        'perSampled': [0,1],#(np.bincount(np.where(samplingResults == teamsPlaying[0])[0], minlength=nsim) + np.bincount(np.where(samplingResults == teamsPlaying[1])[0], minlength=nsim) / nsim).tolist(),

    #        #'perSampled': (np.bincount(np.where(samplingResults == teamsPlaying[0])[0], minlength=nsim) + np.bincount(np.where(samplingResults == teamsPlaying[1])[0], minlength=nsim)).tolist() / nsim,
    #        'winRate': p
    #    })
    

        gameResults = seriesResults,
        
    
    
    return gameResults


def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num
 



#example   

#ATL 2022 vs PHI 2022, 3
#SEA 2022 VS TBR 2022, 3
#STL 2022 VS SDP 2022, 3
#CLE 2022 VS TOR 2022, 3

pred = WS_simulation('ATL', 2022, 'PHI', 2022, 3)
print("ATL vs PHI: "+ most_frequent(pred[0]))

pred = WS_simulation('SEA', 2022, 'TBR', 2022, 3)
print("SEA vs TBR: "+ most_frequent(pred[0]))

pred = WS_simulation('STL', 2022, 'SDP', 2022, 3)
print("STL vs SDP: "+most_frequent(pred[0]))

pred = WS_simulation('CLE', 2022, 'TOR', 2022, 3)
print("CLE vs TOR: "+most_frequent(pred[0]))

#ATL 2022 VS NYM 2022, 5
#LAD 2022 VS STL 2022, 5
#HOU 2022 VS CLE 2022, 5
#NYY 2022 VS SEA 2022, 5

pred = WS_simulation('ATL', 2022, 'NYM', 2022, 5)
print("ATL vs NYM: "+ most_frequent(pred[0]))

pred = WS_simulation('LAD', 2022, 'STL', 2022, 5)
print("LAD vs STL: "+most_frequent(pred[0]))

pred = WS_simulation('HOU', 2022, 'CLE', 2022, 5)
print("HOU vs CLE: "+most_frequent(pred[0]))

pred = WS_simulation('NYY', 2022, 'SEA', 2022, 5)
print("NYY vs SEA: "+most_frequent(pred[0]))


#HOU 2022 VS NYY 2022, 7
#NYM 2022 VS STL 2022, 7

pred = WS_simulation('HOU', 2022, 'NYY', 2022, 7)
print("HOU vs NYY: "+most_frequent(pred[0]))

pred = WS_simulation('NYM', 2022, 'STL', 2022, 7)
print("NYM vs STL: "+most_frequent(pred[0]))

#HOU 2022 VS NYM 2022 7

pred = WS_simulation('NYM', 2022, 'HOU', 2022, 7)
print("NYM vs HOU: "+most_frequent(pred[0]))

