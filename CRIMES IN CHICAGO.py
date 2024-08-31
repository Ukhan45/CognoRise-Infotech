import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
datasets = [
    'Chicago_Crimes_2001_to_2004.csv', 
    'Chicago_Crimes_2005_to_2007.csv', 
    'Chicago_Crimes_2008_to_2011.csv', 
    'Chicago_Crimes_2012_to_2017.csv'
]
df_list = []
for dataset in datasets:
    temp_df = pd.read_csv(dataset, on_bad_lines='skip', engine='python')
    df_list.append(temp_df)
df = pd.concat(df_list, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce') 
df = df.dropna(subset=['Date'])  
df = df.sort_values(by='Date')
df = df[df['Date'] < pd.to_datetime('today') - pd.Timedelta(days=7)]
df.dropna(inplace=True)
print(df.describe())
plt.figure(figsize=(12, 6))
df.groupby(df['Date'].dt.year)['ID'].count().plot()
plt.title('Yearly Crime Trends in Chicago')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.show()
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_crimes = df.groupby('YearMonth').size()
plt.figure(figsize=(12, 6))
monthly_crimes.plot(label='Monthly Crimes')
monthly_crimes.rolling(window=12).mean().plot(label='12-Month Moving Average', color='red')
plt.title('Monthly Crime Trends in Chicago with Moving Average')
plt.xlabel('Year-Month')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()
last_year = df['Date'].dt.year.max() - 1
forecast = df[df['Date'].dt.year == last_year].groupby(df['Date'].dt.month)['ID'].count()
plt.figure(figsize=(12, 6))
forecast.plot(kind='bar', color='orange')
plt.title('Simple Crime Forecast for the Next Year Based on Last Year')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.show()
plt.figure(figsize=(12, 6))
df['Primary Type'].value_counts().plot(kind='bar')
plt.title('Crime Distribution by Type')
plt.xlabel('Crime Type')
plt.ylabel('Number of Crimes')
plt.show()

plt.figure(figsize=(12, 6))
sns.kdeplot(df['Longitude'], df['Latitude'], shade=True, cmap='Reds', bw_adjust=0.5)
plt.title('Crime Density Heatmap in Chicago')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Primary Type', data=df, alpha=0.5, palette='tab10')
plt.title('Crime Locations in Chicago')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()
  
df.to_csv('Crimes_in_Chicago_Cleaned.csv', index=False)
