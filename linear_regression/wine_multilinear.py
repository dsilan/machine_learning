#predict quality of wine with multiple independent values
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset= pd.read_csv('/Users/dorsi/Documents/python/linear_regression/winequality.csv')
#to check if any column contains null value
#print(dataset.isnull().any())
#if there is any true result we should remove nulls
#dataset = dataset.fillna(method='ffill')
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality'].values
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['quality'])
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#regression model has to find the most optimal coefficients for all the attributes
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])

y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))