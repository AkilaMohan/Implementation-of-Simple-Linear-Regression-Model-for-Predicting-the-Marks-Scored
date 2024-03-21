# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
# Dhanush.G
# 2305002006

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:

![Screenshot 2024-03-21 214101](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508176/04f947ba-6faa-481e-9efd-65a9bc57dac0)

![Screenshot 2024-03-21 214114](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508176/a1944f64-aa94-466a-bc7c-2a1cc257d051)

![Screenshot 2024-03-21 214136](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508176/e6b1e44d-cc3d-4d0d-ab4c-a4c6d9f2f0c6)

![Screenshot 2024-03-21 214150](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508176/7ec017f0-9b97-47f6-9f34-e1f19322338c)

![Screenshot 2024-03-21 214205](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155508176/6271653f-c265-4516-82f7-c0fd74faeb08)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
