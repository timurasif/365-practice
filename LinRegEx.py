import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
# import statsmodels.api as sm

raw_data = pd.read_csv('1.04. Real-life example.csv.')

data = raw_data.copy()
# print(data.describe(include='all'))

# Drop 'Model' feature because there are 312 different models, which is very hard for training.
data = data.drop(['Model'], axis=1)
# print(data.head())

# Dropping observations with null values
# print(data.isnull().sum())
data = data.dropna(axis=0)
# print(data.head())



# Removing the outliers

# Show Probability Distribution of 'Price'
# sns.distplot(data['Price'])
# plt.show()

# Find the price value at 99 percentile
val = data['Price'].quantile(0.99)
# Remove Top 1% Price
data = data[data['Price']<val]

# Remove Engine Volume>6.5, since they are unrealistic and dummy values have been added
data = data[data['EngineV']<6.5]

# Find the year below 1 percentile
val = data['Year'].quantile(0.01)
# Remove Top 1% Price
data = data[data['Year']>val]

data = data.reset_index()
# print(data.describe())



# Checking relations between features

# plt.scatter(data['Year'], data['Price'])
# plt.show()
# plt.scatter(data['Mileage'], data['Price'])
# plt.show()

# Price has exponential relation with other features
# Use Log transformation to make a linear relation between price and other features

log_price = np.log(data['Price'])
data['Log Price'] = log_price
data = data.drop(['Price'], axis=1)
# plt.scatter(data['Year'], data['Log Price'])
# plt.show()



# Checking for Multicolinearity among the features

variables = data[['EngineV', 'Mileage', 'Year']]
vif = pd.DataFrame()
vif['Features'] = variables.columns
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

# print(vif)

# Greater vif = Greater multicolinearity
# Usually the cut off value is 5
# Year has vif>10, so we will drop year
data = data.drop(['Year'], axis=1)



# Create dummy variables
data = pd.get_dummies(data, drop_first=True)
# print(data.head())
# print(data.columns.values)

# Rearranging the data, so that the dependant variable comes in the beginning
cols = ['Log Price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
 'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
 'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
 'Registration_yes']

data = data[cols]
#print(data.columns.values)



# Separating targets and inputs
targets = data['Log Price']
inputs = data.drop(['Log Price'], axis=1)

# Scaling the input
scalar = StandardScaler()
scalar.fit(inputs)  # Calculates the mean and standard deviation of the data
inputs_transformed = scalar.transform(inputs)   # Scales the data


# Train Test split
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)


# Training the model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Checking the bias and weights of all features
# print(reg.intercept_)
# print(reg.coef_)


# Testing
result = reg.predict(x_test)
# print(result)



# Analyzing the results

plt.scatter(result, y_test)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.show()

comparison = pd.DataFrame(np.exp(result), columns=['Prediction'])

y_test = y_test.reset_index(drop=True)
comparison['Target'] = np.exp(y_test)

comparison['Difference'] = comparison['Target'] - comparison['Prediction']
comparison['Diff %'] = np.absolute(comparison['Difference']/comparison['Target']*100)
print(comparison.head())
print(comparison.describe())