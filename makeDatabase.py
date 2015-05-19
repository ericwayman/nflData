
import pandas as pd
import numpy as np
import datetime as dt
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
    df.to_sql('rawData', disk_engine, if_exists='append')

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
