#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Speaker-Accent-Recognition
# dataset link : http://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition
# email : amirsh.nll@gmail.com


# In[2]:


import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('dataset.csv')

x = df.drop('language', axis=1)
y = df['language']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf = MLPClassifier(random_state=1, max_iter=1000).fit(x_train, y_train)
clf.predict_proba(x_test)

y_pred = clf.predict(x_test)

print('========== Test Features ============')
print(x_test)

print('============ Predicted Values ===========')
print(y_pred)

clf.score(x_test, y_test)

