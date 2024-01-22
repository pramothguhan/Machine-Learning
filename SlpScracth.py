import numpy as np
class PerceptronScratch:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def train(self, X, Y, n_iteration=10, alpha=0.1, threshold = 0):
        for i in range(n_iteration):
            print("Iteration", i + 1, "\n")
            print("{:<5}{:<5}{:<10}{:<10}{:<10}{:<5}{:<5}".format("X1", "X2", "TARGET", "PREDICTED", "ERROR", "W1", "W2"))
            for x, y in zip(X, Y):
                result = (self.weights[0] * x[0]) + (self.weights[1] * x[1]) + self.bias
                if result > threshold:
                    result = 1
                else:
                    result = 0
                
                error = y - result
                self.weights += alpha * error * x

                print("{:<5}{:<5}{:<10}{:<10}{:<10}{:<5}{:<5}".format(x[0], x[1], y, result, error, self.weights[0], self.weights[1]))

            y_pred = []

            for x, y in zip(X, Y):
                result = (self.weights[0] * x[0]) + (self.weights[1] * x[1]) + self.bias
                y_pred.append(1 if result > threshold else 0)
            
            comparision = Y == y_pred

            print()

            if comparision.all():
                break
model_and = PerceptronScratch(weights=[0.3, -0.2], bias=-0.4)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

model_and.train(x, y)

#AND0001 
#OR0111 
#NAND1110 
#NOR1000 