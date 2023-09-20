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
