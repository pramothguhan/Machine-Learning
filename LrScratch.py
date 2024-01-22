import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:\ML\LR\lr_data.csv")
data.head()
class LinearRegressionScratch:
    def train(self, X, Y):
        self.M = (np.mean(X) * np.mean(Y) - np.mean(X * Y)) / (np.mean(X) * np.mean(X) - np.mean(X * X))
        self.C = np.mean(Y) - self.M * np.mean(X)
    
    def predict(self, X):
        return X * self.M + self.C
model = LinearRegressionScratch()
model.train(data['x'].values, data['y'].values)
plt.scatter(data['x'].values, data['y'].values, c='blue', marker='*')
plt.plot(data['x'].values, model.predict(data['x'].values), c='red', label="Regression Line")
plt.title("Linear Regression Scratch")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
print("Slop:", model.M)
print("Intercept:", model.C)
