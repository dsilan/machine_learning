#task is to predict the maximum temperature 
#taking input feature as the minimum temperature.
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('/Users/dorsi/Documents/python/linear_regression/Weather.csv')
#print(dataset.shape) #number of rows and columns in our dataset
#print(dataset.describe)
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
#plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])
#plt.show()

x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train) #training algorithm
print(regressor.intercept_)
print(regressor.coef_)
#This means that for every one unit of change in Min temperature, 
#the change in the Max temperature is about 0.92%.

#some predictions
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test.flatten(),
'Predicted': y_pred.flatten()})
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='yellow')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='pink')
#plt.show()

plt.scatter(x_test, y_test, color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

#error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))