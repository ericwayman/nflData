import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import sklearn as sk
from sklearn import linear_model
from scipy import stats
from sqlalchemy import create_engine


disk_engine = create_engine('sqlite:///lineData.db')
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