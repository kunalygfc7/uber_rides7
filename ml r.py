import pandas as pd
import numpy as numpy
import pickle
import matplotlib.pyplot as plt


data  = pd.read_csv("taxi.csv")
print(data.head(5))

data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values

print(data_x)
print(data_y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg  = LinearRegression()
reg.fit(X_train,Y_train)

print("Test score : ",reg.score(X_test,Y_test))
print("Train score :",reg.score(X_train,Y_train))


pickle.dump(reg,open("taxi.pkl","wb"))


model = pickle.load(open("taxi.pkl","rb"))
print("Numberofweeklyriders : ",model.predict([[15,1800000,5800,50]]))