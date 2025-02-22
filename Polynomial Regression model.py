
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv(r"E:\NIT Python (ALL Task)\Machine Learning\25th Nov. dataset emp_salary.csv")

X= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

#linear model--linear algorithm(degree-1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#Polynomial model (bydefault degree-2)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


# linear regression visualization
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Linear Regression graph')
plt.xlabellabel('position level')
plt.ylabel('Salary')
plt.show()

# polynomial visualization
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()


#prediction
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred


poly_model_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

























