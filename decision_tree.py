#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/Speaker-Accent-Recognition
# dataset link : http://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition
# email : amirsh.nll@gmail.com


# In[10]:


import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


df = pandas.read_csv('dataset.csv')



lang = {'ES' : 1, 
       'FR' : 2, 
       'GE' : 3, 
       'IT' : 4, 
       'UK' : 5,
       'US' : 6}

df['language'] = df['language'].map(lang)

features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']

x = df[features]
y = df['language']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x, y)


data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)

graph.write_png('dtree.png')
img = pltimg.imread('dtree.png')

implot = plt.imshow(img)

plt.show()


# print(df)

