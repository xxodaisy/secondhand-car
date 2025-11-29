#clustering analysis

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load a data into pandas DataFrame
df = pd.read_csv('final_cars_datasets.csv')

#features to be used for cluster analysis
X = df[['year', 'price', 'engine_capacity', 'mileage']]

#scaling the features if needed to have a similar scale. 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#determine the number of clusters you want to generate and run the cluster algorithm (K-Means) to group the data
#n_cluster with the desired number of clusters

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

#assign the analyzed cluster label to each data in the DataFrame
df['Cluster'] = kmeans.labels_

#evaluate cluster results (optional) -> methods such as elbow method to find the optimal number of clusters, or use cluster evaluation metrics such as Silhouette Score
#save the data that has been added with cluster labels into a csv or excel file for use in viz in PBI

df.to_csv('hasil_kluster.csv', index=False)