import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

data = pd.read_csv('assignment_1_data.csv')

# # region Objective 1: Use the simple linear regression model to fit the data
# y1 = data.iloc[:, 3].to_numpy() # dependence variable is watch_time 
# x1 = data.iloc[:, 5].to_numpy() # indepence variable is confusion

# # x and y were added one axis each to satisfy the data accepted by the regression model in sklearn
# x1 = x1[:, np.newaxis]
# y1 = y1[:, np.newaxis]

# # create the model and output the degree of fit
# model1 = LinearRegression()
# model1.fit(x1, y1)
# predicts = model1.predict(x1)
# R1 = model1.score(x1, y1)
# print('R1 = %.2f' % R1)
# coef = model1.coef_
# intercept = model1.intercept_
# print(coef, intercept)

# # create the figure
# plt.scatter(x1, y1, label='actual value')
# plt.plot(x1, predicts, color='blue', label='predicting value')
# plt.legend()
# plt.show()
# # endregion

# region Objective 2: Use the multiple linear regression model to fit the data
y2 = data.iloc[:, 3].to_numpy() # dependence variable is watch_time 
x2 = data.iloc[:, 4:7].to_numpy() # indepence variables are participation, confusion and key_points

# y was added one axis each to satisfy the data accepted by the regression model in sklearn
y2 = y2[:, np.newaxis]

# create the model and output the degree of fit, slope and intercept
model2 = LinearRegression()
model2.fit(x2, y2)
predicts = model2.predict(x2)
R2 = model2.score(x2, y2)
print('R2 = %.2f' % R2)
coef = model2.coef_
intercept = model2.intercept_
print("model.coef_:", coef)
print("model.intercept_:", intercept)

# endregion 