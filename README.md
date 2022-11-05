# Cricket-Stats - Data Science 
Predicting Results of IPL Matches using Machine Learning

# Firstly let's import packages.

``` 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import array as arr
```

# Adding CSV files Deliveries and Matches

```
deliveries = pd.read_csv('/content/deliveries.csv')
matches = pd.read_csv('/content/matches.csv')

```

# Parse and Preprocess the Data

```
deliveries.head()

matches.head()

```

# Data that contains - null values

```
deliveries.isnull().sum()

matches.isnull().sum()

remove null value -> dropna()

matches.isnull().dropna()

matches.dropna()

```

# Number of Rows and Columns present in Dataset

```
deliveries.shape

matches.shape

deliveries.columns

matches.columns

print('Match played so far: ', matches.shape[0])
print('\n Cities played at: ', matches['city'].unique())
print('\n Team Played at: ', matches['team2'].unique())

matches['id'].max()

data = matches.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h');

```

# Top Batsmen in the Tournament

```
top_players = matches.player_of_match.value_counts()[:1]
fig, ax = plt.subplots()
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top batsmen in the tournament")
sns.barplot(x = top_players.index, y = top_players, orient='v');
plt.show()

```

# Toss Desision

```
toss = matches['toss_winner'] == matches['winner']
plt.figure(figsize=(10,5))
sns.countplot(toss)
plt.show()


plt.figure(figsize=(12,4))
sns.countplot(matches.toss_decision[matches.toss_winner == matches.winner])

```

# Which season had most number of matches?

```
sns.countplot(x='season', data=matches)
plt.show()

```

# Top player of the match Winners

```
top_players = matches.player_of_match.value_counts()[:10]
fig, ax = plt.subplots()
ax.set_ylim([0,20])
ax.set_ylabel("Count")
top_players.plot.bar()
ax.set_title("Top player of the match Winners")
sns.barplot(x = top_players.index, y = top_players, orient='v');
plt.show()

```

# KMeans Clustering 

```
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
x = matches['team1']
y = matches['team2']
plt.scatter(x,y)

```




















