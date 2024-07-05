#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
df=pd.read_csv('Employee-Attrition.csv')
col=df.columns.tolist()
print(col)
df.head()


# In[5]:


df.dtypes


# In[6]:


df.isnull().any()


# In[7]:


df.shape


# In[8]:


df['Department'].unique()


# In[11]:


df['Attrition'].value_counts()


# In[12]:


df.groupby('Attrition').mean()


# In[13]:


df.groupby('Department').mean()


# In[14]:


df.groupby('MonthlyIncome').mean()


# In[16]:


bins = np.linspace(min(df['MonthlyIncome']), max(df['MonthlyIncome']),4)
groupNames = ["low", "med", "high"]
df['SalGroup'] = pd.cut(df['MonthlyIncome'], bins, labels = groupNames, include_lowest = True)
print(df['SalGroup'])


# In[18]:


df.groupby('SalGroup').mean()


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
pd.crosstab(df.Department,df.Attrition).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')


# In[33]:


df=pd.DataFrame(df)
df=df.replace('Yes',1)
df=df.replace('No',0)
df.head()


# In[34]:


table=pd.crosstab(df.SalGroup, df.Attrition)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')


# In[35]:


num_bins = 10
df.hist(bins=num_bins, figsize=(20,15))
plt.savefig("hr_histogram_plots")
plt.show()


# In[39]:


col=['Age','DistanceFromHome',
'Education','EnvironmentSatisfaction','JobLevel','JobSatisfaction','PercentSalaryHike','TotalWorkingYears','YearsSinceLastPromotion']
X=df[col]
y=df['Attrition']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))


# In[40]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))))


# In[42]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7,shuffle=True)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[43]:


from sklearn.metrics import classification_report
print(classification_report(y_test, rf.predict(X_test)))


# In[ ]:




