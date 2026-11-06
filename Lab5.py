
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("/content/data_exp5 (1).csv")

# 2. Select features
X = df[['Age', 'Annual_Income_(k$)', 'Spending_Score']]

# 3. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means
model = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

df['Cluster'] = clusters

# 5. Display cluster profile
print("Cluster Profile:")
print(df.groupby('Cluster')[['Age', 'Annual_Income_(k$)',
                              'Spending_Score']].mean())

# 6. Original data graph
plt.figure(figsize=(8, 5))
plt.scatter(df['Annual_Income_(k$)'], df['Spending_Score'],
            c=clusters, cmap='viridis')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()

# 7. Scaled data graph
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 1], X_scaled[:, 2],
            c=clusters, cmap='viridis')
plt.xlabel("Scaled Annual Income")
plt.ylabel("Scaled Spending Score")
plt.title("Clusters - Scaled Data")
plt.show()
