import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
#import statsmodels.api as sm

data = pd.read_csv('1.02. Multiple linear regression.csv')

#print(data.head())

y = data['GPA']
x = data[['Rand 1,2,3', 'SAT']]


# Scaling the input data
scalar = StandardScaler()
scalar.fit(x)   # Calculates the mean and standard deviation of the data
x_scaled = scalar.transform(x)  # Scales the data


# Creating model
reg = LinearRegression()
reg.fit(x_scaled, y)

# Check the weight of the features that the model calculated
print(reg.coef_)    # SAT will have much higher weight than Rand

# Check the intercept that the model calculated
print(reg.intercept_)


# Prediction
test = pd.DataFrame(data=[[1,1760], [3,1610]], columns=['Rand 1,2,3', 'SAT'])
test_scaled = scalar.transform(test)
result = reg.predict(test_scaled)

print(result)

# Adding the predicted values to the Test dataset
test['Result'] = result
print(test)