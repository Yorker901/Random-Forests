# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:30:46 2022

@author: Mohd Ariz Khan
"""
# Import the data
import pandas as pd
df = pd.read_csv("Fraud_check.csv")

# Converting the Taxable income variable to bucketing. 
df["income"]="<=30000"
df.loc[df["Taxable.Income"] >= 30000,"income"] = "Good"
df.loc[df["Taxable.Income"] <= 30000,"income"] = "Risky"

# Droping the Taxable income variable
df.drop(["Taxable.Income"], axis=1, inplace=True)

df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)

# As we are getting error as "ValueError: could not convert string to float: 'YES'".

# Data Transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = LE.fit_transform(df[column_name])
    else:
        pass
  
# Splitting the data into X and Y variables
X = df.iloc[:,0:5]
Y = df.iloc[:,5]

# Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]

# Data Partition (splitting the data into train and test)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.2)

# Model Fitting
from sklearn.ensemble import RandomForestClassifier 
RFC = RandomForestClassifier(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
RFC.fit(x_train,y_train)

RFC.estimators_
RFC.classes_
RFC.n_features_
RFC.n_classes_

RFC.n_outputs_

RFC.oob_score_  # 74.7833%

# Predictions on train data
y_pred_train = RFC.predict(x_train)

# For accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train,y_pred_train)
# Accuracy ---> 98.75%

np.mean(y_pred_train == y_train)
# Avg. Accuracy ---> 98.75%

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,y_pred_train)

#Prediction on test data
y_pred_test = RFC.predict(x_test)

# Accuracy
test_accuracy = accuracy_score(y_test,y_pred_test)
# Accuracy --> 73.33%

# In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO

tree = RFC.estimators_[5]

dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Creating pdf and png file the selected decision tree
graph.write_pdf('fraudrf.pdf')
graph.write_png('fraudrf.png')

