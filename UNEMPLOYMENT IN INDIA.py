import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

df.columns = df.columns.str.strip()

df = df.drop(columns=['longitude', 'latitude', 'Region.1'])

df['Date'] = df['Date'].str.strip()

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)

print("Columns in df after stripping spaces and dropping irrelevant columns:")
print(df.columns)

print("\nFirst few rows of df:")
print(df.head())

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', label='Unemployment Rate')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend()
plt.show()
