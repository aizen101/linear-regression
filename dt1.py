#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd


# In[27]:


cl=pd.read_csv(r"C:\Users\yadag\OneDrive\Desktop\datasets\Company_Data.csv")
cl.head()


# In[28]:


cl.describe()


# In[29]:


cl.info()


# In[30]:


cl.isnull().sum()


# In[31]:


x=pd.cut(cl["Sales"],bins=[0,7,17],labels=['bad','good'])


# In[32]:


x.head(10)


# In[33]:


cl1=cl.drop("Sales",axis=1)


# In[34]:


cloth=pd.concat([cl,x],axis=1)
cloth.head()


# In[35]:


cloth.isnull().sum()


# In[36]:


x1=cloth.drop("Sales",axis=1)
x1.head()


# In[37]:


y=cloth["Sales"]
y.head()


# In[ ]:





# In[ ]:





# In[38]:


y.isnull().sum()


# In[39]:



y1=y.fillna('good')


# In[48]:


y1.isnull().sum()


# In[59]:


y1.head()


# In[60]:


y2=y1.iloc[:,-1]
y2


# In[41]:


from sklearn.preprocessing import LabelEncoder
fea=['Urban','ShelveLoc','US']
lb=LabelEncoder()
encoded=x1[fea].apply(lb.fit_transform)


# In[42]:


encoded.head()


# In[49]:


x2=x1.drop(['Urban','ShelveLoc','US'],axis=1)


# In[50]:


x_en=pd.concat([x2,encoded],axis=1)


# In[51]:


x_en.head()


# In[61]:


#Splitting the training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_en,y2,test_size=0.2)


# In[62]:


#fitting the model.
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[66]:


#Using Cross validation technique
from sklearn.model_selection import cross_val_score
score=cross_val_score(dt,x_en,y2,cv=5)


# In[67]:


score


# In[68]:


score.mean()


# In[ ]:





# In[63]:


#Training accuracy
dt.score(x_train,y_train)


# In[64]:


#Testing acuuracy
dt.score(x_test,y_test)


# In[ ]:


#Result:-This shows that our model could predict the sales only 70% of the time.


# In[ ]:




