# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Import pandas, numpy and sklearn.                  
Calculate the values for the training data set.             
Calculate the values for the test data set.
Plot the graph for both the data sets and calculate for MAE, MSE and RMSE
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHARMALINGAM S
RegisterNumber:  212223040037
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("CSVs/student_scores.csv")
df.head()
df.tail()
X,Y=df.iloc[:,:-1].values, df.iloc[:,1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(Xtrain,Ytrain)
Ypred=reg.predict(Xtest)
print(Ypred)
plt.scatter(Xtrain,Ytrain,color="orange")
plt.plot(Xtrain,reg.predict(Xtrain),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(Xtest,Ytest,color="blue")
plt.plot(Xtest,reg.predict(Xtest),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("MSE : ",mean_squared_error(Ytest,Ypred))
print("MAE : ",mean_absolute_error(Ytest,Ypred))
print("RMSE : ",np.sqrt(mse))
```

## Output:

![simple linear regression model for predicting the marks scored](sam.png)

![Screenshot 2024-03-02 155658](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/0cee05ae-7270-417d-98bf-28372c8c89e3)

![Screenshot 2024-03-02 155654](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/7a891bbb-f462-4784-b7e7-f91580d50cbd)

![ai ex2](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/aea46236-47d5-4a7d-83e1-daff65870248)

![Screenshot 2024-03-02 160150](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/45a15289-76b7-4f2a-a4d7-95efaffa4491)

![Screenshot 2024-03-02 160841](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/42b45ea8-2a8e-45af-86ef-56b52204ca2d)

![Screenshot 2024-03-02 160936](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/c30fec44-f684-4ea2-8f67-fb7543245022)

![Screenshot 2024-03-02 160957](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/dd1f87be-3de1-4150-9942-bd85660ab845)


![Screenshot 2024-03-02 160354](https://github.com/Dharma23012432/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/ca6a564d-8e06-4a62-8456-355b72e537a8)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
