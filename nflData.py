
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import sklearn as sk
from sklearn import linear_model
from scipy import stats
from sqlalchemy import create_engine


#make the database. All Data files are in a folder called nfllines
disk_engine = create_engine('sqlite:///lineData.db')
start = dt.datetime.now()
for year in range(1978,2014):
    filename = "nfl" + str(year) + "lines.csv"
    df = pd.read_csv('nfllines/'+ filename)
    df = df.rename(columns={c: c.replace(' ','') for c in df.columns}) # Remove spaces from columns
    df = df.drop("TotalLine",axis=1)
    df['Season'] = year
    print '{} seconds: completed {} rows for {}'.format((dt.datetime.now() - start).seconds, len(df.index),str(year))
    #add favorite column to tell you if the hometeam is the favorite
    df["Favorite"] = (df["Line"] > 0).astype(int)
    #Column to record if home team beat the spread"
    df["HomeWinner"] = (df["HomeScore"]- df["VisitorScore"] > df["Line"] ).astype(int)
    #df["HomeWinner"] = (df["HomeScore"]- df["VisitorScore"] > 0 ).astype(int)
    df["VisitorWinner"] = 1 - df["HomeWinner"]

    frame = df[["HomeTeam", "Visitor", "HomeWinner", "VisitorWinner"]]
    frame.columns = [["Home","Visitor","Home", "Visitor"],
                    ["Team","Team","Winner","Winner"]]
    frame.loc[:,("Home","Count")] =1 
    frame.loc[:,("Visitor", "Count")] =1
    frame_stacked = frame.stack(0)
    total_wins = frame_stacked.groupby("Team")["Winner"].cumsum()
    total_games = frame_stacked.groupby("Team")["Count"].cumsum()
    home_win_percentage = total_wins[:,"Home"]/total_games[:,"Home"]
    visitor_win_percentage = total_wins[:,"Visitor"]/total_games[:,"Visitor"]
    frame.loc[:,("Home", "WinPercentage")] = home_win_percentage
    frame.loc[:,("Visitor", "WinPercentage")] = visitor_win_percentage
    frame_restacked = frame.stack(0)
    shifted_win_percentage = frame_restacked.groupby("Team")["WinPercentage"].apply(lambda x: x.shift(1))
    frame.loc[:,("Home", "WinPercentage")] = shifted_win_percentage[:,"Home"]
    frame.loc[:,("Visitor", "WinPercentage")] = shifted_win_percentage[:,"Visitor"]
    wins_In_last_Four = frame_restacked.groupby("Team")["Winner"].apply(lambda x: x.shift(1) + x.shift(2) + x.shift(3) + x.shift(4))
    frame.loc[:,("Home","LastFourGames")] = wins_In_last_Four[:,"Home"]
    frame.loc[:,("Visitor","LastFourGames")] = wins_In_last_Four[:,"Visitor"]

    df["HL4"] = frame[("Home","LastFourGames")]
    df["AL4"] = frame[("Visitor","LastFourGames")]
    df["HomeWinPercentage"] = frame[("Home","WinPercentage")]
    df["AwayWinPercentage"] = frame[("Visitor","WinPercentage")]
    df.to_sql('data', disk_engine, if_exists='append')


#first plot
df = pd.read_sql_query('SELECT Season, HomeScore, VisitorScore, Line FROM data', disk_engine)
spreadDiff = df["HomeScore"] - df["VisitorScore"] - df["Line"]
df["SpreadDifferential"] = spreadDiff
means = df.groupby(["Season"])["SpreadDifferential"].mean()
standErr = df.groupby(["Season"])["SpreadDifferential"].apply(stats.sem)
years = range(1978,2014)
fig, ax = plt.subplots()
ax.errorbar(years, means, yerr = standErr, fmt='o')
ax.set_xlabel('Season')
ax.set_ylabel('Mean with error bars')
ax.set_title('Mean of the score difference  minus the point line by season')
ax.set_xlim(1977, 2014)
ax.axhline(color='black')
fig.savefig("errorbars.pdf")




#some stat analysis
df = pd.read_sql_query('SELECT * FROM data', disk_engine)
df = df.dropna()
dividerYear = 2003
trainingIndices = df.index[df["Season"] < dividerYear]
trainingData = df.loc[trainingIndices,["HomeWinPercentage","AwayWinPercentage","HL4","AL4","Favorite"]]
trainingLabels = np.array(df.loc[trainingIndices,["HomeWinner"]])
logReg = sk.linear_model.LogisticRegression()
logReg.fit(trainingData,trainingLabels)
logReg.fit(trainingData,trainingLabels)

testingIndices = df.index[df["Season"] >= dividerYear]
testingLabels = np.array(df.loc[testingIndices,["HomeWinner"]])
testingFeatures = df.loc[testingIndices,["HomeWinPercentage","AwayWinPercentage","HL4","AL4","Favorite"]]
P = logReg.predict_proba(testingFeatures)

testDF = df[df["Season"] >= dividerYear]
testDF.head(5)
testDF = testDF.dropna()
testDF["PredProbForHomeWin"] = P[:,1]

ProbVals = np.arange(.51,.59,.01)
scores = []
for x in ProbVals:
    testingIndices = testDF.index[(testDF["PredProbForHomeWin"]>=x) | (testDF["PredProbForHomeWin"]<=1-x)]
    testingLabels = np.array(testDF.loc[testingIndices,["HomeWinner"]])
    testingFeatures = testDF.loc[testingIndices,["HomeWinPercentage","AwayWinPercentage","HL4","AL4","Favorite"]]
    score = logReg.score(testingFeatures,testingLabels)
    scores.append(score) 
    print "score = " +str(score)
    print "num samples = " + str(len(testingIndices))

fig2, ax2 = plt.subplots()
width = .8
x = np.arange(len(ProbVals))
ax2.bar(x,scores,width)
ax2.set_xticks(x+ float(width)/2)
ax2.set_xticklabels(ProbVals)
ax2.set_ylabel('Prediction Rate')
ax2.set_xlabel('Lower Bound for estimated probability of predicting the winner')
ax2.set_title('Prediction rate for games with given P value')
fig2.savefig("PredictionRates.pdf")
plt.show
