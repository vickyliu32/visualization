import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import plotly.express as px
from statistics import mean
from plotly.offline import iplot, plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

plt.rcParams['figure.figsize'] =(10,10)
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 10

df = pd.read_csv('spotify-2023.csv')
pd.set_option('display.max_columns', None)
df.head()
df.info()

#check value which is not able to convert to number
for r, v in enumerate(df['streams']):
    try:
        int(v)
    except:
        print('incorrect value')
        print(f'row:{r}')
        print(f'value:{v}')

#drop incorrect value 
df = df.drop(574)

#convert streams from object to number 
df['streams'] = df['streams'].astype('int64')

df['streams'].describe()

pd.unique(df['mode'])
df['mode'].replace(['Major', 'Minor'],[0,1], inplace= True)
df['mode'] = df['mode'].astype('int64')

#display top 15 artists 
artist_count = df['artist(s)_name'].value_counts()
artist_count

#top 10 songs 
top_streams = df[['track_name', 'artist(s)_name', 'streams']].sort_values(by='streams', ascending=False).head(10)
top_streams
plt.figure(figsize=(12,6))
sns.barplot(x=top_streams['streams'],y=top_streams['track_name'], palette='viridis')
plt.xlabel('Streams in billions')
plt.ylabel('Track Name')
plt.title('Top 10 Songs with Most Streams on Spotify')
plt.xticks(rotation=45)
plt.show()

#display relased year songs count
songs_by_year = df['released_year'].value_counts().reset_index()
songs_by_year.columns = ['released_year', 'count']
plt.figure(figsize=(12,6))
sns.barplot(data = songs_by_year.head(14), x='released_year',y='count',palette='viridis')
plt.title =('Number of Songs by Relased Year')
plt.xlabel('Release Year')
plt.ylabel('Number of Songs')
plt.show()

df['released_year'].min()
df['released_year'].max()

#normalize data 
columns = ['streams','released_year','bpm','mode','danceability_%', 
           'valence_%', 'energy_%','acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
df_copy = df[columns].copy()
df_copy.info()

stand_scalar = StandardScaler()
df_copy[columns] = stand_scalar.fit_transform(df_copy[columns])
df_copy.describe()

correlation_matrix = df_copy.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt =".2f")
plt.title('Correlation Matrix')
plt.bbox_inches='tight'
plt.show()

sns.pairplot(df_copy, x_vars = columns, y_vars =['streams'])
plt.show()





