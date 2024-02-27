# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## Name: Anbuselvan.S
## Reference No: 212223240008
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Anbuselvan.S
RegisterNumber: 212223240008

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
print(df)
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
print(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![Screenshot 2024-02-27 092736](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841744/062893b4-5b85-4b74-9cdb-cad3729a3ac0)
![Screenshot 2024-02-27 092802](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841744/db551c74-55e5-4901-a62e-35e5debdd1af)
![Screenshot 2024-02-27 092820](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139841744/15b5ea8a-ce32-4a4d-84e5-6cdeeaeef863)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
