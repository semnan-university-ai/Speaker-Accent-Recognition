#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Speaker-Accent-Recognition
# dataset link : http://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition
# email : amirsh.nll@gmail.com


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('dataset.csv')


x = df.iloc[:, 1:] # features data
y = df.iloc[:, :1] # class data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)   # split train data and validation (test) data with test data including 20 percent of whole data


# ========== (Begin) This block of code perform normalizing or scaling the features which is a good practice so that all of features can be uniformly evaluated
# scaler = StandardScaler()
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# ========== (End)  Feature scaling

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print('========== Test Features ============')
print(x_test)

print('============ Predicted Values ===========')
print(y_pred)


# ============ Evaluating the algorithm

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('\n\n============================   Comparing Error rate with different k value ============================= \n\n')

error = []
# Calculating error for K values between 1 and 10
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    pred_i=pred_i.reshape(len(x_test),1)
    error.append(np.mean(pred_i != y_test))
    
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[ ]:




