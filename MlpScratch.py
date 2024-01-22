import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
data = pd.read_csv(r'D:\ML\MLP\disease.csv')
data.head()
cats = ['cp', 'restecg', 'slope', 'ca', 'thal']

for col in cats:
    data = data.join(pd.get_dummies(data[col], prefix=col))

data.drop(columns=cats, inplace=True)
data.head()
X = data.drop('target', axis=1)
y = pd.DataFrame(data, columns=['target'])
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
class MultilayerPerceptron:
    def __init__(self, random_state=None):
        np.random.seed(seed=random_state)
        self.w1 = np.random.normal(size=(27, 10))
        self.w2 = np.random.normal(size=(10, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def cross_entropy(self, y, output):
        return np.mean(- y * np.log(output) - (1 - y) * np.log(1 - output))

    def feedforward(self):
        self.z2 = self.input.dot(self.w1)
        self.a2 = self.sigmoid(self.z2)

        self.z3 = self.a2.dot(self.w2)
        self.output = self.sigmoid(self.z3)

    def backprop(self):
        self.output_error = self.output - self.y
        self.output_delta = self.output_error * self.sigmoid_deriv(self.output)

        self.a2_error = self.output_delta.dot(self.w2.T)
        self.a2_delta = self.a2_error * self.sigmoid_deriv(self.a2)

        self.w2 -= self.learning_rate * self.a2.T.dot(self.output_delta)
        self.w1 -= self.learning_rate * self.input.T.dot(self.a2_delta)


    def train(self, x, y, epochs=1000, lr=0.1):
        self.learning_rate = lr
        self.loss = []
        
 
        self.input = x
        self.y = y
        self.output = np.zeros(y.shape)

        for _ in range(1, epochs + 1):

            self.feedforward()
            self.backprop()

            self.loss.append(self.cross_entropy(self.y, self.output))

    def predict(self, x):
        self.input = x
        self.feedforward()

    def evaluate(self, y, threshold=0.5):
        self.y = y
        self.y_hat = np.where(self.output > threshold, 1, 0)
        print(f'Confusion Matrix\n{confusion_matrix(self.y, self.y_hat)}\n')
model = MultilayerPerceptron(random_state=42)
model.train(X_train, y_train, epochs=3000, lr=0.01)
model.predict(X_test)
model.evaluate(y_test, threshold=0.2)