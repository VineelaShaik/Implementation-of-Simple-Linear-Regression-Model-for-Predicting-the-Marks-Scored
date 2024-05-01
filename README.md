# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.<br>
2.Assign hours to X and scores to Y.<br>
3.Implement training set and test set of the dataframe.<br>
4.Plot the required graph both for test data and training data.<br>
5.Find the values of MSE , MAE and RMSE. <br>

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vineela Shaik
RegisterNumber:  212223040243
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import  LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
### df.head()<br>
![Screenshot 2024-03-06 131705](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/be18d4d7-50ac-4f69-b705-e469ab86f367)<br>
### df.tail()<br>
![Screenshot 2024-03-06 131823](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/7b9924ae-a129-454f-9ee4-3a43a3e0d545)<br>
### x=df.iloc[:,:-1].values<br>
### x<br>
![Screenshot 2024-03-06 131932](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/cf5914cf-6b24-4840-81d4-ff0421c2a852)<br>
### y=df.iloc[:,1].values<br>
### y<br>
![Screenshot 2024-03-06 132129](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/1842d97f-f5bb-4e43-aebc-ddc99b777069)<br>
### y_pred<br>
![Screenshot 2024-03-06 132248](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/2d9c0a55-ab22-4345-8d5e-83b3f6f09195)<br>
### y_test<br>
![Screenshot 2024-03-06 132322](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/f899f52e-a8bb-466a-8287-29d0e24e9975)<br>
### Training set<br>
![Screenshot 2024-03-06 132458](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/63009a10-5133-4252-8f0d-53ba0f44587a)<br>
### Testing set<br>
![Screenshot 2024-03-06 132549](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/f36735a7-fd1e-42ea-8193-8c0c480931b3)<br>
### Errors<br>
![Screenshot 2024-03-06 132626](https://github.com/VineelaShaik/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144340862/0e3fe82c-19e5-4679-b2ef-86349e1904c7)<br>
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
