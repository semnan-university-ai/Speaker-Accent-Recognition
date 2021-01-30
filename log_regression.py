#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Speaker-Accent-Recognition
# dataset link : http://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition
# email : amirsh.nll@gmail.com


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

# x = df.drop('language', axis=1)
# y = df['language']


df = pd.DataFrame(df,columns= ['language','X1', 'X2','X3','X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12'])

# print(df)

x = df[['X1', 'X2','X3','X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']]
y = df['language']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

log_regression = LogisticRegression(max_iter=5000)
log_regression.fit(x_train, y_train)

y_pred = log_regression.predict(x_test)

# confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# sn.heatmap(confusion_matrix, annot=True)
print('========== Test Features ============')
print(x_test)

print('============ Predicted Values ===========')
print(y_pred)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

# plt.show()

