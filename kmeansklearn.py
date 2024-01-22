import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:\ML\K-Means\income.csv")
data.head()
X = data[["Age", "Income($)"]].values
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
clusters = model.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red")
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()