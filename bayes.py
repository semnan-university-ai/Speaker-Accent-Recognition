#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Speaker-Accent-Recognition
# dataset link : http://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition
# email : amirsh.nll@gmail.com


# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df = pd.read_csv('dataset.csv')

x = df.drop('language', axis=1)
y = df['language']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)





# ===================== Gaussian Naive Bayes ==================== #
model = GaussianNB()
model.fit(x_train, y_train)

print('========== Test Features ============')
print(x_test)

y_pred = model.predict(x_test)
print('============ Predicted Values ===========')
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy : ', accuracy)

