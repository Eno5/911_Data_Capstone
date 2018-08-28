import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plt.show() after all matplotlib plots

df = pd.read_csv('911.csv')

df.info()
print(df.head(4))

# Top 5 zipcodes
print(df['zip'].value_counts().head(5))

# Top 5 townships
print(df['twp'].value_counts().head(5))

# Unique values in title column
print(df['title'].nunique())

# New column (Reason) with values EMS, Fire, and Traffic based on title column
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

sns.countplot(x='Reason',data=df)
plt.show()

# Convert time string to datetime
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# Create Hour, Month, and Day of Week columns from timeStamp
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.weekday()) #Mon - Sun, 0-6

# dictionary for mapping
dmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(arg=dmap)

# Plot number of calls by DoW and Month
sns.countplot(x='Day of Week',hue='Reason',data=df)
plt.show()

sns.countplot(x='Month',hue='Reason',data=df)
plt.show()

# groupby Month
byMonth = df.groupby(by='Month').count()

byMonth['timeStamp'].plot()
plt.show()

# linear regression of monthly calls
sns.lmplot(x='Month', y='timeStamp', data=byMonth.reset_index())
plt.show()

# Create Date column
df['Date'] = df['timeStamp'].apply(lambda x: x.date())

# groupby Date
df.groupby(by='Date').count()['timeStamp'].plot()

# restructure data to have DoW as rows and Hour as columns
heat = df[['timeStamp','Day of Week','Hour']].groupby(by=['Day of Week','Hour']).count().unstack()

sns.heatmap(data=heat)
plt.show()