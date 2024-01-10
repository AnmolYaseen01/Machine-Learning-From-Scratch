
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print(train.head)

print(train.info())
print(test.info())

print(train.describe())
print(test.describe())

"""removing the missing values in the training data set as follow:"""

train=train.dropna()

"""to check if the dataset contains any duplicates"""

duplicates_exist=train.duplicated().any()
print(duplicates_exist)

x_train=train['x']
y_train=train['y']
x_test=test['x']
y_test=test['y']
print(x_train.shape)
print(x_test.shape)

"""we have a one-dimensional array which may lead to unexpected behavior. So, we will reshape the above to (699,1) and (300,1) to explicitly specify that we have one label per data point."""

x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)

print(x_train.min(), x_train.max())

"""standardization"""

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled.min(), x_train_scaled.max())

plt.scatter(x_train, y_train)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

"""creating a linear regression model"""

model=LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
mse=mean_squared_error(y_test, predictions)
print(mse)

"""plotting the regression line:"""

plt.plot(x_test, predictions, color='red',linewidth=2, label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

