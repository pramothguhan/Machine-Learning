import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:\Pramoth\MIT Documents\Sem 6\ML\Lab\ML lab\LR\lr_data.csv")
data.head()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data['x'].values.reshape(-1, 1), data['y'].values)
plt.scatter(data['x'].values, data['y'].values, c='blue', marker='*')
plt.plot(data['x'].values, model.predict(data['x'].values.reshape(-1, 1)), c='red', label="Regression Line")
plt.title("Linear Regression Scikit Learn")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
print("Intercept:", model.intercept_)
print("Slop:", model.coef_)

