#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


fraud=pd.read_csv(r"C:\Users\yadag\OneDrive\Desktop\p\Fraud_check.csv")
fraud.head()


# In[3]:


fraud.isnull().sum()


# In[4]:


fraud.shape


# In[5]:


from sklearn.preprocessing import LabelEncoder
fr_feautures=['Undergrad','Marital.Status','Urban']
lb=LabelEncoder()
encoded=fraud[fr_feautures].apply(lb.fit_transform)


# In[11]:


encoded.head()


# In[13]:





# In[7]:


fr=fraud.drop(['Urban','Marital.Status','Undergrad'],axis=1)
fr.head()


# In[9]:


fraud1=pd.concat([fr,encoded],axis=1)
fraud1.head()


# In[12]:


fraud1['inc'] = pd.cut(x=fraud['Taxable.Income'], bins=[0,30000,100000],labels=['Risky','good'])


# In[13]:


fraud1.head()


# In[14]:


x=fraud1.drop(['inc'],axis=1)
x.head()


# In[16]:


y=fraud1.iloc[:,-1]
y.head()


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)


# In[25]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[21]:


y_pred=classifier.predict(x_test)


# In[ ]:


#Result:The accuracy is pretty high.


# In[ ]:




