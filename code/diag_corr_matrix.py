from string import letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
# rs = np.random.RandomState(23)
# d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                  columns=list(letters[:26]))

all_data = pd.read_csv('combined2.csv')

# Compute the correlation matrix
corr = all_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(285, 8, as_cmap=True)
# 220, 10 are standard red-blue




# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=20, yticklabels=20,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
# g.set_xticklabels(rotation=30)
plt.xticks(rotation=30)
plt.yticks(rotation=30)

# finding groups of columns that have the most negative or most positive correlations
# I will then delete these groups and see how it affects my RMSE
# first I will group them by a >0.7 or <-0.7, from EDA, that looks like a good choice

answer = pd.DataFrame()
answer['lists_of_similar_cols'] = range(143)

for col in range(len(corr)):
    list_of_cols = corr.ix[col,:]
    list_at_threshold = []
    for i, value in list_of_cols.iteritems():
        if value > 0.6 or value <-0.6:
            list_at_threshold.append(i)
    answer['lists_of_similar_cols'][col] = list_at_threshold


print answer



plt.show()
