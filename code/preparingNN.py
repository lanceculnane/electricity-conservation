#so... the NN on the actual raw data didn't yield anything
# so I'm going to do some feature engineering here and then see if a NN
# or a random forest will do the best job of predicting electricity
# savings based on the raw electricity usage itself 15-days prior to the
# start of the competition

'''
#brainstorming: Each proposed column will have a #)
*** section 1 ***
# 1) first 15-min electricity usage
2) last 15-min electricity usage
3) percentage decrease from 1-2 (as a positive #)
4-12) repeat 1-3 for 3 more time-scales (avging 1st 5 days and last 5 days etc)

*** section 2 ***
13) overall trend- straight line- fit lin reg and make a prediction 30 days out perc decrease ('-' will mean an increase is expected)
(maybe 1 column for raw predicted decrease, and then another where it is
1- (1st 5 day avg - predicted decrease / 1st day 5 avg)  )
or... maybe just (1 - (y from equation on predicted 30th day of comp or last day)/ (y from equation on first day)
14) R^2 value?
I looked at higher order regressions but they dont seem to fit well

*** section 3 *** peaks
Take a MA10 to determine the peak usage amount (write a program which finds the value where on either side the value is lower and also the value is much higher than the overall average so it is not in the noise which occurs at night....)
 There should be about 10 peaks for each data set maybe a couple more or less

 15 - 20) give raw values? or normalized to avg value...?
 21-26) create some columns showing a trendline with just these data points and predicting how % decrecrease in future?
 27 - 33) make some dummy variables - yes -no are the final 5 peaks lower than the first 5 peaks? are the final 5 peaks decreasing?

 *** section 4 *** MAs! moving averages!

 34 - 70) make lots of MAs (MA10, MA50, MA100, MA 200, MA 300 etc) and make columns with their values or normalized values at some check-points (maybe at midnight of each day or the value at each peak from section 3...)

 70 -100) can now make lots of dumy variables based on info from 34-70... yes-no is the MA10 increasing in the last 20 hrs, 50 hrs? same Q for each MA. Is the MA increasing 10 days back but decreasing 5 days back? do this for each MA. The NN or forest may find some interesting combos that I cant see, like: if the electricity long MA400 is trending up near the 15th day but trending down with MA50, then there is a better chance the school will have a high % savings of electricity etc.


 I think all of this will give better results with a regression NN or random forest because it will be able to look at all 100 features and figure out patterns- some features will contribute more than others in the final answer...

 also im wondering... should i just first make 1-2 features dealing with the magnitude in case that info is important, and then StANDARDIZE all data for features 3-100 since we want everything in terms of % decrease anyway...yeah i think so...or I could divide all values by the max value *100 so that the highest point for everyone is 100- yeah I like that... and then in the same step i can just automatically make that max value the 1st feature in case the overall magnitude matters

 i mentioned the peaks being important but the troughs could be also (the electricity usage at night) perhaps i should make a feature with the ratio between the peak and the top of the trough- big schools with a huge difference in these might have more of a chance of decreasing electricity since you cant do much with the night-time elecetricity (backgroud electricity)- we could also look at trends of the background electricity...

... i should also consider getting rid of all weekend data...
 models within models!'''

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import decimal
import pylab as pl

start = datetime.datetime.now().time()


'''SECTION 1: looking at percent decrease from beginning to end'''
orig_X = pd.read_csv('all67train.csv', header=None) # each school is in a column and there are 67 columns and 1440 rows of timeseries data
orig_X_T = orig_X.T   # transforming it all columns (each school) to rows, as it will be used in new df
new = pd.DataFrame()
del_orig_X = pd.read_csv('all67train_deleted.csv', header=None)
del_orig_X_T = del_orig_X.T
# this is the same data but with all weekends deleted, and it also has the peaks all starting at about the 50th row.
# Note: I'm going to use this data for the linear regression features and moving average features
# it doesn't have 1440 rows, but 1003 rows

new["mean"] = np.mean(orig_X) # creating new column where each school (in row) has its mean electricity usage in the column
new["del_mean"] = np.mean(del_orig_X)

norm_X = orig_X/new["mean"] # original data normalized to its mean
norm_X_T = norm_X.T
# now I will continue to build out of the df named 'new' but using norm_X or norm_X_T as the source (and del_norm_X for lin reg and MAs)
del_norm_X = del_orig_X/new["del_mean"]
del_norm_X_T = del_norm_X.T


new["first"] = norm_X_T[0]   #first 15-min normalized value
new["last"] = norm_X_T[1439] #last 15-min normalized value
new["f_l_perc_dec"] = ((norm_X_T[0] - norm_X_T[1439])/norm_X_T[0])*100
     # this returns percent decrease from first 15 min to last 15 min where a decrease will be positive
new["avg_f_two_days"] = np.mean(norm_X[:192]) #192 because indexing goes up to 192 and only includes 191 but thats ok because indexing starts at 0
    # average electricity usage for first 2 days. 4*24*2 = 192 15-min chunks in 2 days
new["avg_l_two_days"] = np.mean(norm_X[1247:])
    # average electricity usage for last 2 days
new["two_day_perc_dec"] = ((new["avg_f_two_days"] - new["avg_l_two_days"])/new["avg_f_two_days"])*100
    # % decrease from avg of first 2 days to avg of last 2 days where a decrease is positive and an increase is negative
'''# repeating same idea for first and last 5 days and then first and last 7.5 days (ie 'half')'''
new["avg_f_five_days"] = np.mean(norm_X[:480])
new["avg_l_five_days"] = np.mean(norm_X[959:])
new["five_day_perc_dec"] = ((new["avg_f_five_days"] - new["avg_l_five_days"])/new["avg_f_five_days"])*100
new["avg_f_half"] = np.mean(norm_X[:720])
new["avg_l_half"] = np.mean(norm_X[719:])
new["halves_perc_dec"] = ((new["avg_f_half"] - new["avg_l_half"])/new["avg_f_half"])*100

ma10 = pd.read_csv("ma10.csv")
ma50 = pd.read_csv("ma50_del.csv")
ma100 = pd.read_csv("ma100_del.csv")
ma200 = pd.read_csv("ma200_del.csv")
ma300 = pd.read_csv("ma300_del.csv")
ma400 = pd.read_csv("ma400_del.csv")

'''SECTION 2: linear regression on first 15 days'''

# lr = linear_model.LinearRegression()
# test_X = np.array(range(1440))
# test_X_use = test_X[:,np.newaxis]
# test_y = norm_X[2].values
# test_y_use = test_y[:,np.newaxis]
# lr.fit(test_X_use,test_y_use)
# lr.predict(test_X_use)
# # cross_val_predict returns an array of the same size as `y` where each entry
# # is a prediction obtained by cross validation:
# # predicted = cross_val_predict(lr, test_X, test_y)
#
#
# # Plot outputs
# plt.scatter(test_X_use, test_y_use,  color='purple')
# plt.plot(test_X_use, lr.predict(test_X_use), color='blue',
#          linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
'''it's a bit messy- but I know that the above works and I can put any number from 0-66 after the norm_X and plot a graph of the data and its simple linear regression best-fit line. I will now use that to create a function which will do that where I will input which index to use (0-66) and it will output the intercept and coefficient which I can use to predict electricity 30 days in the future....

  It could be interesting for the coef, intercept, and r2 to each form a new feature in my df called 'new' (as well as some various predictions), but if I use a function which spits out multiple things it will be difficult for each of those things to form a new column...perhaps the return of the function can write directly to an empty spot in the column, like:
   for i in range(67):
       new.ix[i,new["coef"]], new.ix[i,new["intercept"]], new.ix[i,new["r_squared"]], new.ix[i,new["30_prediction"]] = lin_reg_func(i) where the func returns: lr.coef_, lr.intercept_, lr.score(X,y), mx+b where x is about 2880

       ...and after all that is finished, it will be easy to take those values and make another column with % decrease from the original slope intercept to the final predicted value

       hmmm... im looking at the data and it looks like sometimes the coefficient (slope of the line) is a good predictor of electricity conservation, but it looks like it depends on which competition it was in, so I think the NN should also have a feature 1-9 which labels which competition it was in....'''

''' for future reference, here I practiced adding to a df with funcs:

In [21]: def test_func2(i):
    ...:     return float(i)**2/2
    ...:

In [22]: def test_func(i):
    ...:     return float(i)**2

In [25]: def multi(i):
    ...:     return test_func(i), test_func2(i)

In [51]: for i in range(40):
    ...:     three, four = multi(i)
    ...:     test.ix[i, 'third_col'] = three
    ...:     test.ix[i, 'fourth_col'] = four
--- and then it successfully makes a 3rd and 4th column with the correct values in each row!
'''

def run_lr(i):
    '''INPUT: i, index of a row of data that has a length of 1440
       OUTPUT: lr.coef_, lr.intercept_, lr.score(X,y), y from mx+b where x is 2880'''
    lr = linear_model.LinearRegression()
    test_X = np.array(range(1003))
    test_X_use = test_X[:,np.newaxis]
    test_y = del_norm_X[i].values
    test_y_use = test_y[:,np.newaxis]
    lr.fit(test_X_use,test_y_use)
    lr.predict(test_X_use)
    y = lr.coef_*2880 + lr.intercept_
    return lr.coef_, lr.intercept_, lr.score(test_X_use,test_y_use), y

# for i in range(67):
#     co, inter, rsq, y_value = run_lr(i)
#     new.ix[i, 'coef'] = co
#     new.ix[i, 'intercept'] = inter
#     new.ix[i, 'rsquared'] = rsq
#     new.ix[i, '30prediction'] = y_value # predicts electricity usage 30 days after competition begins

'''SECTION 3: finding peaks with Moving Average (MA)'''

# ma10 = pd.DataFrame() #create empty SMA10 dataframe
# ma10["index"] = range(1440)
# ma50 = pd.DataFrame() #create empty SMA10 dataframe
# ma50["index"] = range(1003)

def make_MA10(i):
    '''INPUT: column i from norm_X
       OUTPUT: simple 10 moving average'''
    transform = norm_X[i]
    for j in range(1440):
        if j < 10:
            ma10.ix[j, 'sma10_{}'.format(i)] = None
        else:
            ma10.ix[j, 'sma10_{}'.format(i)] = np.mean(transform[j-10:j])

# for i in range(67):
#     make_MA10(i)

def make_MA50(i):
    '''INPUT: column i from norm_X
       OUTPUT: simple 50 moving average'''
    transform = del_norm_X[i]
    for j in range(1003):
        if j < 50:
            ma50.ix[j, 'sma50_{}'.format(i)] = None
        else:
            ma50.ix[j, 'sma50_{}'.format(i)] = np.mean(transform[j-50:j])

# for i in range(67):
#     make_MA50(i)

def find_peaks(i):
    '''INPUT: column i from norm_X
       OUTPUT: returns 10 peak heights'''
    school = ma50.ix[:,i+1]  # brings up each school (i) ma50
                                #+1 because there are two index columns
    school_mean = np.mean(school)
    count = 0 # for each school, we will count the peaks
    for j in range(1003):
        if school[j] > school_mean and school[j-1] > school[j-2] and school[j-1] > school[j] and school[j-2] > school[j-3] and school[j-3] > school[j-4] and school[j-4] > school[j-5] and school[j-5] > school[j-6] and school[j-6] > school[j-7] and school[j-7] > school[j-8] and school[j-8] > school[j-9] and school[j-9] > school[j-10] and count < 10:
            count += 1
            new.ix[i,"peak_{}".format(count)] = school[j]
# I have to make sure that 10 in a row are going up to the peak- because on some peaks, there are 5 small peaks which go up and down and I only want 1
# it works well now, but 2 (index 16 and 57) have 11 peaks and the rest have 10 so i'm going to only take 10
# for i in range(67):
#     find_peaks(i)

'''SECTION 4: MA's !!!'''

'''so... now I have 23 columns (features) including 10 peaks from the MA50 to play with and for a NN to crunch on. I have MA10 and MA50 matrices made. Now I'd like to make MA100, MA200, MA300, MA400 and lots of dummy variables from as many logical and illogical ideas as I can come up with (or have the patience to make). In general, I think the more data the better and even if things don't make sense, its possible that a NN or random forest (RF) WILL find it helpful. For instance, maybe a particular yes (1), no (0) dummy condition doesn't matter at all, EXCEPT a RF may find it helpful in spliting up one particular node within a subset of data- then it has served a purpose even if it is not known to a human like me.'''

#first, make MA100, MA200, MA300, and MA400
# ma100 = pd.DataFrame() #create empty SMA100 dataframe
# ma100["index"] = range(1003)
# ma200 = pd.DataFrame() #create empty SMA200 dataframe
# ma200["index"] = range(1003)
# ma300 = pd.DataFrame() #create empty SMA300 dataframe
# ma300["index"] = range(1003)
# ma400 = pd.DataFrame() #create empty SMA400 dataframe
# ma400["index"] = range(1003)

def make_MA100(i):
    '''INPUT: column i from norm_X
       OUTPUT: simple 100 moving average'''
    transform = norm_X[i]
    for j in range(1003):
        if j < 100:
            ma100.ix[j, 'sma100_{}'.format(i)] = None
        else:
            ma100.ix[j, 'sma100_{}'.format(i)] = np.mean(transform[j-100:j])

def make_MA200(i):
    '''INPUT: column i from norm_X
       OUTPUT: simple 200 moving average'''
    transform = norm_X[i]
    for j in range(1003):
        if j < 200:
            ma200.ix[j, 'sma200_{}'.format(i)] = None
        else:
            ma200.ix[j, 'sma200_{}'.format(i)] = np.mean(transform[j-200:j])

def make_MA300(i):
    '''INPUT: column i from norm_X
       OUTPUT: simple 300 moving average'''
    transform = norm_X[i]
    for j in range(1003):
        if j < 300:
            ma300.ix[j, 'sma300_{}'.format(i)] = None
        else:
            ma300.ix[j, 'sma300_{}'.format(i)] = np.mean(transform[j-300:j])

def make_MA400(i):
    '''INPUT: column i from norm_X
       OUTPUT: simple 400 moving average'''
    transform = norm_X[i]
    for j in range(1003):
        if j < 400:
            ma400.ix[j, 'sma400_{}'.format(i)] = None
        else:
            ma400.ix[j, 'sma400_{}'.format(i)] = np.mean(transform[j-400:j])
# for i in range(67):
#     make_MA100(i)
#     make_MA200(i)
#     make_MA300(i)
#     make_MA400(i)

#I'm going to make the ma's their own files since it takes awhile to make them
# oh good! Now it only takes 7 sec instead of 10 min

# ma10.to_csv("ma10.csv")
# ma50.to_csv("ma50_del.csv")
# ma100.to_csv("ma100_del.csv")
# ma200.to_csv("ma200_del.csv")
# ma300.to_csv("ma300_del.csv")
# ma400.to_csv("ma400_del.csv")


'''NOTE! I had to create an extra 'index' column the way I made all the MA's... but I didn't have to do that on the norm_X. So each column in the MA's is +1- remember that while coding etc!'''
'''
# making a column which is the average of the all 10 peaks
new["avg_peaks"] = new[["peak_1", "peak_2", "peak_3", "peak_4", "peak_5", "peak_6", "peak_7", "peak_8", "peak_9", "peak_10"]].mean(axis=1)
# making a column which is the average of the first 5 peaks
new["avg_l_peaks"] = new[["peak_6", "peak_7", "peak_8", "peak_9", "peak_10"]].mean(axis=1)
# avg last 5 peaks
new["avg_f_peaks"] = new[["peak_1", "peak_2", "peak_3", "peak_4", "peak_5"]].mean(axis=1)
# percent decrease from first 5 peaks to last 5 peaks
new["perc_decrease_peaks"] = ((new["avg_f_peaks"] - new["avg_l_peaks"])/new["avg_f_peaks"])*100
# new column: was the last 5 peaks < first 5 peaks? 1 for yes
'''
'''ok... I'm getting a little stressed out with how complicated this is getting and I'm running out of time, so here's my plan for the last bit, and then I'm going to run it on NN and RF and add more features if the results are terrible:
1) new col with MA100 value at row 650
2) new col with MA100 value at row 950
3) new 1 if #2 < #1, 0 otherwise
4) for each MA: create a col at row 500, 650, 800, 950- thats going to be 4 cols per MA, and we have MA10, MA50, MA100, MA200, MA300, MA400... so 24 new cols for the NN or RF to use for pattern recognition (well 2 of the cols i already made... Ill do this #4 first, then #3)
I'm looking at my data briefly, and it looks like MA50, MA100, and MA300 is most-interesting...so...
5) new col: enter 1 if MA300-900 is higher than MA300-1000 and also higher than MA300-650; 0 otherwise (did the trend peak and now is starting to go down)
6) new col yes-no: was MA100 > MA300 at row650 and now at row950 MA100<MA300
7) oh! i just realized, i might as well make a stddev column- I did mean of the entire original datafram, but not the stddev... maybe 1 std of the original data, and 1 col that is std of the standardized data
Then I'm done!... i wonder if I should create a feature from when the NN ran on the raw data?... maybe later if the results are shitty'''


'''
new["MA10_500"] = ma10.ix[500,3:].values
new["MA10_650"] = ma10.ix[650,3:].values
new["MA10_800"] = ma10.ix[800,3:].values
new["MA10_950"] = ma10.ix[950,3:].values
new["MA50_500"] = ma50.ix[500,3:].values
new["MA50_650"] = ma50.ix[650,3:].values
new["MA50_800"] = ma50.ix[800,3:].values
new["MA50_950"] = ma50.ix[950,3:].values
new["MA100_500"] = ma100.ix[500,2:].values
new["MA100_650"] = ma100.ix[650,2:].values
new["MA100_800"] = ma100.ix[800,2:].values
new["MA100_950"] = ma100.ix[950,2:].values
new["MA200_500"] = ma200.ix[500,2:].values
new["MA200_650"] = ma200.ix[650,2:].values
new["MA200_800"] = ma200.ix[800,2:].values
new["MA200_950"] = ma200.ix[950,2:].values
new["MA300_500"] = ma300.ix[500,2:].values
new["MA300_650"] = ma300.ix[650,2:].values
new["MA300_800"] = ma300.ix[800,2:].values
new["MA300_950"] = ma300.ix[950,2:].values
new["MA400_500"] = ma400.ix[500,2:].values
new["MA400_650"] = ma400.ix[650,2:].values
new["MA400_800"] = ma400.ix[800,2:].values
new["MA400_950"] = ma400.ix[950,2:].values
new["MA300_900"] = ma300.ix[900,2:].values
new["MA300_1000"] = ma300.ix[1000,2:].values

new["MA100_decreasing"] = 0
for i in range(67):
    if new["MA100_950"][i] < new["MA100_650"][i]:
        new["MA100_decreasing"][i] = 1
#interesting... almost all of them are 0; not decreasing

new["MA300_bending_down"] = 0
for i in range(67):
    if new["MA300_900"][i] > new["MA300_650"][i] and new["MA300_900"][i] > new["MA300_1000"][i]:
        new["MA300_bending_down"][i] = 1
#only one chart satisfies this...

new["MA200_bending_down"] = 0
for i in range(67):
    if new["MA200_650"][i] > new["MA200_500"][i] and new["MA200_650"][i] > new["MA200_950"][i]:
        new["MA200_bending_down"][i] = 1

new["MA100_cross_MA300"] = 0
for i in range(67):
    if new["MA100_650"][i] > new["MA300_650"][i] and new["MA300_950"][i] > new["MA100_950"][i]:
        new["MA100_cross_MA300"][i] = 1

new["std"] = np.std(orig_X, ddof=1)
new["std_norm"] = np.std(norm_X, ddof=1)
'''

if __name__ == '__main__':

    # print norm_X.head()
    # print norm_X.tail()
    # print new.head()
    # print ma10.tail()

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, axisbg='white')
    # fig.subplots_adjust(top=.85)
    ax.set_title('15-Day, Normalized Baseline Electricity Usage')
    ttl = ax.title
    ttl.set_position([.5, 1.05])
    # sns.distplot(y, color='blue')
    ax.set_xlabel('Days (each dot = 15 min)')
    ax.set_ylabel('1 = mean electricity usage')




    # print run_lr(15)
    # plt.scatter(pl.frange(0.0104166666,15,0.010416666666), norm_X.ix[:,0],  color='red')
    plt.scatter(pl.frange(0.0104166666,10.44791666,0.010416666666), del_norm_X.ix[:,0],  color='red')
    plt.plot(pl.frange(0.0104166666, 10.44791666, 0.01041666666), ma50.ix[:,2].T, color='blue',
             linewidth=3)
    print len(ma50.ix[:,2])
    # plt.scatter(range(1440),norm_X.ix[:,42], color = 'purple')
    # plt.plot(range(1440),ma10.ix[:,43], color = 'blue')
    plt.plot(pl.frange(0.0104166666, 10.44791666, 0.01041666666),ma100.ix[:,2], color = 'purple')
    plt.show()
    print "started at: ", start
    print "finished at: ", datetime.datetime.now().time()
    # print ma400.tail()
    # new.to_csv("new.csv")
    # print new.shape
