import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:\Pramoth\MIT Documents\Sem 6\ML\Lab\ML lab\K-Means\income.csv")
data.head()
X = data[["Age", "Income($)"]].values
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()
def distance(point_1, point_2):
    return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)

class KMeansScratch:    
    def train(self, X, K=2, n_iter=100):
        self.centroids = X[np.random.choice(range(X.shape[0]), replace = False, size = K)]
        
        for _ in range(n_iter):
            clusters = [[] for _ in range(K)] 
            
            for point in X:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                clusters[np.argmin(distances)].append(point)
            
            self.centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

    def predict(self, X):
        result = []

        for point in X:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]            
            result.append(np.argmin(distances))
        
        return result
model = KMeansScratch()
model.train(X, K=3, n_iter=100)
clusters = model.predict(X)
clusters
model.centroids
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], c="red")
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()
