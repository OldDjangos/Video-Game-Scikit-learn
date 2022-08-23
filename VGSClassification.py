#!/usr/bin/env python
# coding: utf-8

# In[426]:


import numpy as np
import pandas as pd


# In[427]:


data = pd.read_csv("vgsales.csv")


# In[428]:


data.head()


# In[429]:


data.info()


# In[430]:


data.dropna(inplace=True)


# In[431]:


data.isnull().sum()


# In[432]:


np.unique(data["Platform"])


# In[433]:


data['Platform'].replace('2600', 'PC', inplace=True)
np.unique(data["Platform"])


# In[434]:


label1 = data['Platform'].unique().tolist()
data['Platform'] = data['Platform'].apply(lambda n: label1.index(n))
np.unique(data['Platform'])


# In[435]:


label2 = data['Genre'].unique().tolist()
data['Genre'] = data['Genre'].apply(lambda n: label2.index(n))
np.unique(data['Genre'])


# In[436]:


label3 = data['Publisher'].unique().tolist()
data['Publisher'] = data['Publisher'].apply(lambda n: label3.index(n))
np.unique(data['Publisher'])


# In[437]:


data.head()


# In[438]:


data["SuccOrNot"] = (data["Global_Sales"]>0.22) #1->success 0->unseccess


# In[439]:


data['SuccOrNot']


# In[440]:


x = data[['Platform','Genre','Publisher','NA_Sales','Other_Sales']]
y = data["SuccOrNot"]


# In[441]:


#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
from sklearn.preprocessing import StandardScaler
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)


# In[442]:


from sklearn.linear_model import LogisticRegression


# In[443]:


LR = LogisticRegression()


# In[444]:


LR.fit(x_train,y_train)


# In[445]:


y_predict = LR.predict(x_test)
LR.coef_


# In[446]:


LR.intercept_


# In[447]:


#Evaluation
from sklearn.metrics import classification_report,confusion_matrix


# In[448]:


print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test,y_predict,labels=[1,0],target_names=["Success","Unsuccess"]))


# In[449]:


from sklearn.model_selection import cross_val_score,GridSearchCV


# In[450]:


LR_CV = cross_val_score(LR, x, y, cv=5, scoring='accuracy')


# In[463]:


print("Accuracy:\n",LR_CV.mean()*100,"%")


# In[453]:


#RandomForest
from sklearn.ensemble import RandomForestClassifier


# In[454]:


RFC = RandomForestClassifier()


# In[455]:


RFC.fit(x_train,y_train)


# In[456]:


y_predict2 = RFC.predict(x_test)


# In[457]:


print(confusion_matrix(y_test, y_predict2))
print(classification_report(y_test,y_predict2,labels=[1,0],target_names=["Success","Unsuccess"]))


# In[458]:


#RandomForest cv
RFC_CV = RandomForestClassifier()
param_dict = {"n_estimators": [120, 200, 300, 500, 800],
              "max_depth":[5,8,10]} 
RFC_CV = GridSearchCV(RFC_CV, param_grid=param_dict, cv=3)


# In[459]:


RFC_CV.fit(x_train,y_train)


# In[462]:


#Evaluate
#Best parameters
print("Best parameters:\n",RFC_CV.best_params_)
##Best score
print("Best score:\n", RFC_CV.best_score_)
#Best_estimator_
print("Best estimator:\n", RFC_CV.best_estimator_)


# In[464]:


#GRAPH
name_list =['LR','LR_CV','RF','RF_CV']
num_list = [0.92,LR_CV.mean(),0.94,0.93]
plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
plt.show()  


# In[ ]:




