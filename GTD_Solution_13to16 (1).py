
# coding: utf-8

# ## Predictions on Global Terrorist Database 2013 to 2016

# ## Importing the libraies

# In[98]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# In[99]:


df = pd.read_excel('C:/Users/ddddd/Downloads/Gtd/gtd_13to16_0617dist.xlsx')
print(df)


# In[100]:


df.columns


# In[101]:


df


# In[102]:


df.shape


# In[103]:


df.isnull().sum(axis=0)


# In[104]:


data_set = df.dropna(subset=[
'approxdate'  ,
'resolution'  ,
'provstate'  ,
'city'  ,
'latitude'  ,
'longitude'  ,
'specificity'  ,
'location'  ,
'summary'  ,
'alternative'  ,
'alternative_txt'  ,
'propextent'  ,
'propextent_txt'  ,
'propvalue'  ,
'propcomment'  ,
'ishostkid'  ,
'nhostkid'  ,
'nhostkidus'  ,
'nhours'  ,
'ndays'  ,
'divert'  ,
'kidhijcountry'  ,
'ransom'  ,
'ransomamt'  ,
'ransomamtus'  ,
'ransompaid'  ,
'ransompaidus'  ,
'ransomnote'  ,
'hostkidoutcome'  ,
'hostkidoutcome_txt'  ,
'nreleased'  ,
'addnotes'  ,
'scite1'  ,
'scite2'  ,
'scite3' ,
'related','eventid', 'iyear', 'country', 'region', 'crit1', 'crit2', 'crit3', 'doubtterr', 'attacktype1', 'suicide', 'claimed', 'success', 'targtype1', 'gname', 'weaptype1'])


# In[105]:


df.claimed.unique().tolist()


# In[106]:


#df.dtypes


# In[107]:


#Feature Selection or Feature Engineering
data_set = df[['eventid', 'iyear', 'country', 'region', 'crit1', 'crit2', 'crit3', 'doubtterr', 'attacktype1', 'suicide', 'claimed', 'success', 'targtype1', 'gname', 'weaptype1']]


# In[108]:


data_set.head(10)


# In[109]:


data_set = data_set.values


# In[110]:


#data_set['gname'] = data_set['gname'].astype('category')


# In[111]:


#data_set['weaptype1'] = data_set['weaptype1'].astype('category')


# In[112]:


#data_set['attacktype1'] = data_set['attacktype1'].astype('category')


# In[113]:


#data_set['targtype1'] = data_set['targtype1'].astype('category')


# ## Target stored in y

# In[114]:


y=data_set[['claimed']]


# In[ ]:


y.head(100)


# ## Taking features to predict claimed

# In[ ]:


Gtd_features = ['crit1','crit2','crit3','doubtterr',
        'suicide','success']


# In[ ]:


X = data_set[Gtd_features].copy()


# In[ ]:


X.columns


# In[ ]:


y.columns


# ## Train Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# In[ ]:


type(X_train)
type(X_test)
type(y_train)
type(y_test)
X_train.head()
y_train.describe()


# ## Fit on Train set

# In[ ]:


terrorism_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
terrorism_classifier.fit(X_train, y_train)


# In[ ]:


type(terrorism_classifier)


# ## Predict on Test set

# In[ ]:


predictions = terrorism_classifier.predict(X_test)


# In[ ]:


predictions[:20]


# In[ ]:


y_test['claimed'][:10]


# ## Measuring the accuracy of classifier

# In[ ]:


accuracy_score(y_true = y_test, y_pred = predictions)

