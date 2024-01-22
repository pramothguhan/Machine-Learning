import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:/Pramoth/MIT Documents/Sem 6/ML/Lab/ML lab/data.csv")
data.head()
X = data.drop("target",axis=1)
Y = data["target"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
X_train.shape,y_train.shape
X_test.shape,y_test.shape
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train,y_train)
plot_tree(model);
predictedResults = model.predict(X_test)
actualResults = np.array(y_test)
performance = confusion_matrix(actualResults,predictedResults)
print(performance)
pZeroCount = list(predictedResults).count(0)
pOneCount = list(predictedResults).count(1)
aZeroCount = list(actualResults).count(0)
aOneCount = list(actualResults).count(1)
plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

ax1.bar(["Actual","Predicted"],[aZeroCount,pZeroCount])
ax1.set(ylabel="Count",title="For label Zero")

ax2.bar(["Actual","Predicted"],[aOneCount,pOneCount])
ax2.set(ylabel="Count",title="For label One");
accuracy_score(actualResults,predictedResults)