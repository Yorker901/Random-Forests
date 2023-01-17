# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 18:41:02 2022

@author: Mohd Ariz Khan
"""
# Import the data
import pandas as pd
df = pd.read_csv("Company_Data.csv")  # read the data
df
df.head()

# Get information of the dataset
df.info()
df.shape
print('The shape of our data is:', df.shape)
df.isnull().any()

# EDA (Exploratory Data Analysis)

# let's scatter plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data = df, hue = 'Sales')

# Creating dummy vairables dropping first dummy variable
df = pd.get_dummies(df, columns=['Urban','US'], drop_first=True)
df
df.info()

# Mapping
df['ShelveLoc'] = df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})
df.head()

# Split the variables
X = df.iloc[:,1:10] # Independent Variable
X

Y = df['Sales']   # Target Variable
Y

# Counts in Target variable
df.Sales.value_counts()
col = list(df.columns)
col

import numpy as np
# Labels are the values we want to predict
labels = np.array(df['Income'])

# Remove the labels from the data & axis 1 refers to the columns
features  = df.drop('Income', axis = 1)

# Saving feature names for later use
features_list = list(df.columns)

# Convert to numpy array
features  = np.array(df)

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# Shape of Train Features, Test Features, Train Labels, Test Labels
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

# Model Fitting 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5) 
regressor.fit(X_train, Y_train)

print("Node counts:",regressor.tree_.node_count)
print("max depth:",regressor.tree_.max_depth)

Y_pred_train = regressor.predict(X_train) 
Y_pred_test = regressor.predict(X_test) 

from sklearn.metrics import mean_squared_error
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))


from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=500,max_depth=8,max_features=0.7,
                      max_samples=0.6,random_state=10)

RFR.fit(X_train, Y_train)
Y_pred_train = RFR.predict(X_train) 
Y_pred_test = RFR.predict(X_test) 
Training_err = mean_squared_error(Y_train,Y_pred_train).round(2)
Test_err = mean_squared_error(Y_test,Y_pred_test).round(2)

print("Training_error: ",Training_err.round(2))
print("Test_error: ",Test_err.round(2))


