#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


db=pd.read_csv(r"C:\Users\yadag\OneDrive\Desktop\p\Diabetes.csv")
db.head()


# In[3]:


db.shape


# In[4]:


db.isnull().sum()


# In[9]:


x=db.iloc[:,:-1]
x.head()


# In[10]:


y=db.iloc[:,-1]
db.head()


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8)


# In[13]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)


# In[16]:


y_pred = classifier.predict(x_test)


# In[17]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


#result=as we can see the accuracy of the model is pretty low


# In[ ]:




