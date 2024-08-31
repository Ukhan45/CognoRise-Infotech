import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv('marketing_campaign.csv', delimiter='\t')

print(df.head())

print(df.describe())

print(df.isnull().sum())

df.columns = df.columns.str.strip()

df['Dt_Customer'] = pd.to_datetime(df.get('Dt_Customer'), errors='coerce')

df = df.dropna()  

df = pd.get_dummies(df, drop_first=True)

numeric_df = df.select_dtypes(include=[np.number])

scaler = StandardScaler()
scaled_df = scaler.fit_transform(numeric_df)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_df)

pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', data=pca_df)
plt.title('PCA of Customer Data')
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(pca_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 7))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_clusters = 3  
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(pca_df)

pca_df['Cluster'] = clusters

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('Customer Segments')
plt.show()

df['Cluster'] = clusters

for i in range(optimal_clusters):
    print(f"Cluster {i} statistics:")
    print(df[df['Cluster'] == i].describe())
    print("\n")

df.to_csv('new_file.csv', index=False)
