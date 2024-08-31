import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Customertravel.csv'  
df = pd.read_csv(file_path)

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, color='skyblue', bins=10)
plt.title('Distribution of Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['ServicesOpted'], kde=True, color='lightgreen', bins=6)
plt.title('Distribution of Services Opted', fontsize=16)
plt.xlabel('Services Opted', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
sns.countplot(x='FrequentFlyer', data=df, hue='FrequentFlyer', palette='Set2', legend=False)
plt.title('Frequent Flyer Status', fontsize=16)
plt.xlabel('Frequent Flyer', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.subplot(2, 2, 2)
sns.countplot(x='AnnualIncomeClass', data=df, hue='AnnualIncomeClass', palette='Set3', legend=False)
plt.title('Annual Income Class', fontsize=16)
plt.xlabel('Annual Income Class', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.subplot(2, 2, 3)
sns.countplot(x='AccountSyncedToSocialMedia', data=df, hue='AccountSyncedToSocialMedia', palette='Set1', legend=False)
plt.title('Account Synced To Social Media', fontsize=16)
plt.xlabel('Account Synced To Social Media', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.subplot(2, 2, 4)
sns.countplot(x='BookedHotelOrNot', data=df, hue='BookedHotelOrNot', palette='Set1', legend=False)
plt.title('Booked Hotel or Not', fontsize=16)
plt.xlabel('Booked Hotel Or Not', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='ServicesOpted', y='Age', data=df, palette='coolwarm', hue='ServicesOpted', dodge=False)
plt.title('Age vs Services Opted', fontsize=16)
plt.xlabel('Services Opted', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.show()

plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
sns.countplot(x='FrequentFlyer', hue='Target', data=df, palette='Set2')
plt.title('Churn by Frequent Flyer Status', fontsize=16)
plt.xlabel('Frequent Flyer', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.subplot(2, 2, 2)
sns.countplot(x='AnnualIncomeClass', hue='Target', data=df, palette='Set3')
plt.title('Churn by Annual Income Class', fontsize=16)
plt.xlabel('Annual Income Class', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.subplot(2, 2, 3)
sns.countplot(x='AccountSyncedToSocialMedia', hue='Target', data=df, palette='Set1')
plt.title('Churn by Account Synced To Social Media', fontsize=16)
plt.xlabel('Account Synced To Social Media', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.subplot(2, 2, 4)
sns.countplot(x='BookedHotelOrNot', hue='Target', data=df, palette='Set1')
plt.title('Churn by Booked Hotel Or Not', fontsize=16)
plt.xlabel('Booked Hotel Or Not', fontsize=14)
plt.ylabel('Count', fontsize=14)

plt.tight_layout()
plt.show()
