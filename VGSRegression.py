#!/usr/bin/env python
# coding: utf-8

# In[456]:


import numpy as np
import pandas as pd


# In[457]:


data = pd.read_csv("vgsales.csv")


# In[458]:


data.head()


# In[459]:


data.info()


# In[460]:


data.isnull().any()


# In[461]:


#处理缺失值
data.dropna(inplace=True)


# In[462]:


data.isnull().sum()


# In[463]:


np.unique(data["Platform"])


# In[464]:


data['Platform'].replace('2600', 'PC', inplace=True)
np.unique(data["Platform"])


# In[465]:


label1 = data['Platform'].unique().tolist()
data['Platform'] = data['Platform'].apply(lambda n: label1.index(n))
np.unique(data['Platform'])


# In[466]:


label2 = data['Genre'].unique().tolist()
data['Genre'] = data['Genre'].apply(lambda n: label2.index(n))
np.unique(data['Genre'])


# In[467]:


label3 = data['Publisher'].unique().tolist()
data['Publisher'] = data['Publisher'].apply(lambda n: label3.index(n))
np.unique(data['Publisher'])


# In[468]:


data.head()


# In[469]:


#筛选特征值和目标值
x = data[['Platform','Genre','Publisher','NA_Sales','EU_Sales']]
y = data["Global_Sales"]


# In[470]:


y.head()


# In[471]:


x.head()


# In[491]:


#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[492]:


from sklearn.preprocessing import StandardScaler
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)


# In[493]:


x_train


# In[494]:


#LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
# 4)estimator
estimator = LinearRegression()
estimator.fit(x_train,y_train)
# 5)get model
print("coef_:\n",estimator.coef_)
print("intercept_:\n",estimator.intercept_)
# 6)evaluate
y_predict = estimator.predict(x_test)
squared_error = mean_squared_error(y_test,y_predict)
LR_r2 = r2_score(y_test,y_predict)
print("mean_squared_error:\n",squared_error)
print("R2:\n",LR_r2)
LR_accuracy_on_test = estimator.score(x_test, y_test)  
print("Accuracy on test: ", LR_accuracy_on_test* 100,"%")
fig = plt.figure(figsize=(10,6))


# In[495]:


#LinearRegression CV
from sklearn.model_selection import cross_val_predict
estimator2 = LinearRegression()
cv_value = cross_val_predict(estimator2, x, y, cv=10)
LRCV_r2 = r2_score(y,cv_value)
print("R2:\n",LRCV_r2)


# In[496]:


#KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
# 4)estimator
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train,y_train)
# 6)evaluate
y_predict = knn.predict(x_test)
squared_error = mean_squared_error(y_test,y_predict)
KNN_r2 = r2_score(y_test,y_predict)
print("mean_squared_error:\n",squared_error)
print("R2:\n",KNN_r2)
KNN_accuracy_on_test = knn.score(x_test, y_test)  
print("Accuracy on test: ", KNN_accuracy_on_test)
fig = plt.figure(figsize=(10,6))


# In[502]:


#KNN_CV
KNN_CV = KNeighborsRegressor(n_neighbors=5)
KNN_CV_r2 = -cross_val_score(KNN_CV, x, y, cv=5,scoring='r2')
KNN_CV_r2 = (KNN_CV_r2.mean())/100
print("KNN_R2:\n",KNN_CV_r2)


# In[498]:


import matplotlib.pyplot as plt


# In[499]:


#GRAPH
name_list =['LR','LR_CV','KNN','Knn_CV']
num_list = [LR_r2,LRCV_r2,KNN_r2,KNN_CV_r2]
plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
plt.show()  


# In[ ]:




