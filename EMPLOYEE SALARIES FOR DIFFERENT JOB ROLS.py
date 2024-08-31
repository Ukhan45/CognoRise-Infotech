import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ds_salaries.csv')

print("Dataset Information:")
df.info()

print("\nFirst 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.histplot(df['salary_in_usd'], kde=True)
plt.title('Distribution of Salaries')
plt.xlabel('Salary (USD)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='job_title', y='salary_in_usd', data=df)
plt.title('Salary Distribution by Job Title')
plt.xticks(rotation=90)
plt.show()

avg_salary_by_title = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
print("\nAverage Salary by Job Title:")
print(avg_salary_by_title)

numeric_df = df.select_dtypes(include=['number'])  
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='experience_level', y='salary_in_usd', data=df)
plt.title('Salaries by Experience Level')
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(x='company_location', y='salary_in_usd', data=df)
plt.title('Salaries by Company Location')
plt.xticks(rotation=90)
plt.show()

if 'work_year' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='work_year', y='salary_in_usd', data=df, marker='o')
    plt.title('Salary Trends Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Salary (USD)')
    plt.show()

if 'employment_type' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='employment_type', y='salary_in_usd', data=df)
    plt.title('Salaries by Employment Type')
    plt.show()

if 'remote_ratio' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='remote_ratio', y='salary_in_usd', data=df)
    plt.title('Salaries by Remote Ratio')
    plt.show()
