# in all variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import  mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-50]
diabetes_X_test = diabetes_X[-50:]


diabetes_y_train = diabetes.target[:-50]
diabetes_y_test = diabetes.target[-50:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train) #  training the data

diabetes_y_predicted = model.predict(diabetes_X_test)

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)


#plt.scatter(diabetes_X_test, diabetes_y_test)
#plt.plot(diabetes_X_test, diabetes_y_predicted)

#plt.show()

#Mean squared error is:  1789.5698810318188
#Weights:  [   1.19680495 -233.50366519  519.88450429  304.48097981 -726.44619916
#  415.80697199   83.01061117  203.1648289   667.92936385  105.27983988]
#Intercept:  152.2312643406476

