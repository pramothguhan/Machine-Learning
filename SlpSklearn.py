import numpy as np
from sklearn.linear_model import Perceptron
model_and = Perceptron()

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

model_and.fit(x, y)
model_and.predict(x)






#AND0001 
#OR0111 
#NAND1110 
#NOR1000 