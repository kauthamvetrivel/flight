#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[3]:


data=pd.read_csv("/Users/ks/Downloads/Flight  - Sheet1.csv")
data.head()


# In[20]:


plt.figure(figsize=(8,6))
sns.catplot(x="Length",y="Breadth",data=data)


# In[21]:


plt.boxplot(data.Length.dropna())


# In[22]:


plt.boxplot(data.Breadth)


# In[11]:


plt.subplot(1,2,1)
plt.boxplot(data.Breadth)
plt.subplot(1,2,2)
plt.boxplot(data.Distance)


# In[12]:


data.hist()


# In[13]:


sns.countplot(x="Distance",data=data)


# In[23]:


from scipy import stats
stats.shapiro(data.Distance)


# In[15]:


stats.ttest_rel(data.Distance, data.Breadth)


# In[27]:


data.fillna(data.Length.median())


# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
clf.predict(X)


# In[52]:


y = data.Distance
x = data.iloc[0:,[2,3]]
from sklearn.model_selection import train_test_split
trainx,testx = train_test_split(x,test_size = 0.2)
trainy,testy = train_test_split(y,test_size = 0.2)


# In[53]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(trainx, trainy)
clf.score(testx, testy)


# In[43]:


clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[44]:


X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)


# In[45]:


clf.fit(X, y)


# In[ ]:




