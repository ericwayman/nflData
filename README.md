# nflData
#Script to implement a Logistic regression model of the form
#Y^*_i = b_0+ b_1*HWP_i + b_2*AWP_i + b_3*HL4_i + b_4*AL4_i b_5FAV_i + e_i
#i is the index of the game number
#Y^*_i is our prediction for Y_i the indicator of the home team beating the spread in game i
#{b_0,b_1,b_2,b_3,b_4,b_5} the parameters to train
#HWP_i: overall win percentage (relative to the spread) of home team
#AWP_i: overall win percentage (relative to the spread) of away team
#HL4_i: number of times home team has beaten the spread in the last 4 games played (this season)
#HL4_i: number of times away team has beaten the spread in the last 4 games played (this season)
#FAV_i: indicator the home team is the favorite 
