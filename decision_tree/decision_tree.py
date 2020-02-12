import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
#loading the data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("../../diabetes.csv", header=None, names= col_names)
pima.head()

#feature selection
#dependent = target & independent = feature
feature_cols= ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
x = pima[feature_cols]
y = pima.label

#splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#building decision tree model
decisionTreeClassifier = DecisionTreeClassifier() #create Decision Tree classifer object
decisionTreeClassifier = decisionTreeClassifier.fit(x_train, y_train) #train the classifier
y_pred = decisionTreeClassifier.predict(x_test) #predict for test data

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
