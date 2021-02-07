import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#We use a simple favorite data set for Linear Regression code.
df = pd.DataFrame({"age": [22,25,47,52, 46,56,55,60,62,61,18,28,27,29,49,55,25,58,19,18,21,26,40,45,50,54,23], "bought_insurance":[0,0,1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0]})


#Spliting data set in two part, training and testing
#We have a dataset with 27 units and we leaves 7 units as test set.
test = df.sample(7)

train = df[~df.isin(test)]
train.dropna(inplace = True)

#Activation function definition
def sigmoid(x):
  return 1/(1+np.exp(-x))

#Loss function definition
def square_loss(y_pred, target):
  return np.mean(pow((y_pred - target),2))

X_tr, y_tr = train.age, train['bought_insurance']
X_te, y_te = test.age, test['bought_insurance']

#Model setup
lr = 0.01
W = np.random.uniform(0,1)
b = 0.1

for i in range(1000):
  z = np.dot(X_tr , W) + b


y_pred = sigmoid(z)
l = square_loss(y_pred, y_tr)
gradient_W = np.dot((y_pred - y_tr).T, X_tr)/X_tr.shape[0]
gradient_b = np.mean(y_pred - y_tr)
W = W - lr * gradient_W
b = b - lr * gradient_b

#Test the performance of medel
for i in range(len(X_te)):
  r = sigmoid(np.dot(X_te , W) + b)
# We compute performance for test set units
print(r)