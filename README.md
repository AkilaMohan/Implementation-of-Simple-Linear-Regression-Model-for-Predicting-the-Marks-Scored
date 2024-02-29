# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import necessary libraries (e.g., pandas, numpy,matplotlib).
2.Load the dataset and then split the dataset into training and testing sets using sklearn library.
3.Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4.Use the trained model to predict marks based on study hours in the test dataset.
5.Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sharon Harshini L M
RegisterNumber: 212223040193

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train
y_pred

plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
 
```

## Output:
![3](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/7fff9222-db4e-4dfe-9a4f-398f8cc8f97a)
![1](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/b27187d9-dc0b-4beb-8bde-ef4223aa112f)
![2](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/361974fa-d1c8-4acf-887c-95711160c7fa)
![4](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/77214238-7b9f-4c50-8414-6658e1351035)
![5](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/7281cf7f-1455-4b0f-aeaf-b0a6bd98bdeb)
![6](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/f963e516-275e-4099-aea2-b2a82f7d9702)
![7](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149555539/3b565252-c3ce-4c12-9ba7-c4df4631960f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
