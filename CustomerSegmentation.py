import pandas as pd
#Reading dataset
retail_df = pd.read_csv('https://confrecordings.ams3.digitaloceanspaces.com/Retail_1.csv')
#Performing EDA
print(retail_df.head())
print(retail_df.isnull().sum())
retail_df = retail_df.dropna()
print(retail_df.shape)
# new column: Monetary 
retail_df['Monetary'] = retail_df['Quantity']*retail_df['UnitPrice']
grouped_df = retail_df.groupby('CustomerID')['Monetary'].sum()
grouped_df = grouped_df.reset_index()
print(grouped_df.head())

# new column :frequency
frequency = retail_df.groupby('CustomerID')['InvoiceNo'].count()
frequency = frequency.reset_index()
frequency.columns = ['CustomerID','frequency']
print(frequency.head())

# new column :recency
# convert to datetime
retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'], format='%d-%m-%Y %H:%M')
# compute the diff
retail_df['diff'] = max(retail_df['InvoiceDate']) - retail_df['InvoiceDate']

# new column: recency
last_purchase = retail_df.groupby('CustomerID')['diff'].min()
last_purchase = last_purchase.reset_index()
print(last_purchase.head())

# merge the two dfs
grouped_df = pd.merge(grouped_df, frequency, on='CustomerID', how='inner')
print(grouped_df.head())
# merge the third dataframe
rfm = pd.merge(grouped_df, last_purchase, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Monetary', 'frequency', 'recency']
# number of days only
rfm['recency'] = rfm['recency'].dt.days
print(rfm.head())
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# instantiate
scaler = StandardScaler()
rfm_df_scaled = scaler.fit_transform(rfm[['Monetary','frequency','recency']])
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Monetary', 'frequency', 'recency']
rfm_df_scaled.head()

#Training the KMeans with n_clsuters =3 
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)
rfm_df_scaled['cluster_id'] = kmeans.labels_
print(rfm_df_scaled.head())

#Plotting the clustered graph
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rfm_df_scaled['frequency'], rfm_df_scaled['recency'], rfm_df_scaled['Monetary'],
           linewidths=1, alpha=.5,
           edgecolor='black',
           s = 100,
           cmap='spring',
           c=rfm_df_scaled['cluster_id'])
ax.set_xlabel("Frequency")
ax.set_ylabel("Recency")
ax.set_zlabel("Amount")
plt.show()
