import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import sklearn as sk
import scipy.stats
from sklearn import linear_model
from scipy import stats
from sqlalchemy import create_engine
from scipy.stats import binom
 

#helper function to compute confidence intervals
def binom_interval(total, p=.5,confint=0.95):
    quantile = (1 - confint) / 2.
    lower = binom.ppf(quantile, total, p)
    upper = binom.ppf(1 - quantile, total,p)
    return [lower, upper]


#load data
disk_engine = create_engine('sqlite:///lineData.db')

df = pd.read_sql_query('SELECT * FROM data', disk_engine)
df = df.dropna()

#Choose divider year.  Train logifistic regression on all years before dividerYear starting from 1978
#test on all years after until 2013
dividerYear = 2003
trainingIndices = df.index[df["Season"] < dividerYear]
trainingData = df.loc[trainingIndices,["HomeWinPercentage","AwayWinPercentage","HL4","AL4","Favorite"]]
trainingLabels = np.array(df.loc[trainingIndices,["HomeWinner"]]).ravel()
logReg = sk.linear_model.LogisticRegression()
logReg.fit(trainingData,trainingLabels)

testingIndices = df.index[df["Season"] >= dividerYear]
#total number of games in the testing set
numGames = len(df.index)
#predict whether or not the home team wins
testingLabels = np.array(df.loc[testingIndices,["HomeWinner"]])
testingFeatures = df.loc[testingIndices,["HomeWinPercentage","AwayWinPercentage","HL4","AL4","Favorite"]]
P = logReg.predict_proba(testingFeatures)

testDF = df[df["Season"] >= dividerYear]
testDF = testDF.dropna()
testDF["PredProbForHomeWin"] = P[:,1]

ProbVals = np.arange(.51,.59,.01)
scores = []
standErr = []
#create file to produce summary of prediction strategy
f = open('./predictionSummary.txt','w')
f.write("Summary of tests when betting on games where the predicted probability of success is above a threshold. \n \n")
f.write("|Threshold Probability | Proportion of Games  |  Number of Games  |  Success Rate  |  P Value (p=.5)  |  95%% CI\n")
for x in ProbVals:
    testingIndices = testDF.index[(testDF["PredProbForHomeWin"]>=x) | (testDF["PredProbForHomeWin"]<=1-x)]
    testingLabels = np.array(testDF.loc[testingIndices,["HomeWinner"]])
    testingFeatures = testDF.loc[testingIndices,["HomeWinPercentage","AwayWinPercentage","HL4","AL4","Favorite"]]
    sampleSize = len(testingIndices)
    score = logReg.score(testingFeatures,testingLabels)
    scores.append(score)
    pValue = scipy.stats.binom_test(score*sampleSize, sampleSize, p=0.5)*.5 
    #print("Right tailed P value: %f" %pValue)
    preds = logReg.predict(testingFeatures)
    correctPreds =  testingLabels.ravel()-preds + 1
    correctPreds[correctPreds != 1] =0
    standErr.append(stats.sem(correctPreds))
    #print "score = " +str(score)
    #print "num samples = " + str(sampleSize)
    CI = binom_interval(sampleSize,.5,0.95)
    CI = np.array(CI)/float(sampleSize)
    f.write("       %.3f           |        %.3f         |        %i       |       %.2f     |    %.3f  |   (%.2f,%.2f)   \n" 
        %(x,sampleSize/float(numGames), sampleSize,score,pValue,CI[0],CI[1]))

f.close() 

#Plot prediction rates with error bars
fig, ax = plt.subplots()
width = .8
x = np.arange(len(ProbVals))
ax.bar(x,scores,width)
ax.errorbar(x+float(width)/2, scores, yerr = standErr, fmt='o',color = 'red')
ax.set_xticks(x+ float(width)/2)
ax.set_xticklabels(ProbVals)
ax.set_ylabel('Prediction Rate')
ax.set_xlabel('Lower Bound for estimated probability of predicting the winner')
ax.set_title('Prediction rate for games with given success probability')
fig.savefig("PredictionRates.pdf")
plt.show