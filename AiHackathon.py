#!/usr/bin/env python
# coding: utf-8
# In[1]:
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

# ### This is the beginning of the notebook 
# In[167]:
df = pd.read_csv('train.csv')

# In[168]:
df.head()

# ### Exploratory stage: 
# 
# In[14]:
# Looking at the data types
df.dtypes

# In[169]:
# get day of the week
empty_sent_dayofweek = []

sent_list = pd.to_datetime(df['sent_time'])

for i in range(len(list(df.apt_date))):
    empty_sent_dayofweek.append(sent_list[i].dayofweek)

# In[171]:
df['sent_dayofweek'] = empty_sent_dayofweek

# In[172]:
sent_dayofweek_list = list(df.sent_dayofweek)

empty_sent_weekend = []

for i in range(len(list(df.apt_date))):
    if sent_dayofweek_list[i] >= 5:
        empty_sent_weekend.append(1)
    else:
        empty_sent_weekend.append(0)
    
df['sent_weekend'] = empty_sent_weekend
# In[173]:

df.head()

# In[174]:
df.drop(['clinic', 'sent_dayofweek', 'ReminderId', 'pat_id', 'family_id'], axis=1,inplace = True)

# In[175]:

df.head()

# ### What we need to clean: 
# - apt_date, sent_time and send_time should be in datetime format 
# - Convert cat columns to dummies
# - age bins? 
# - drop reminder ID
# - convert date time format
# In[176]:
sent_list = list(df.sent_time)

# In[177]:
empty_1 = []
for i in range(len(list(df.apt_date))):
    index_val = sent_list[i].find(' ')
    empty_1.append(sent_list[i][:index_val])

# In[178]:
df.sent_time = empty_1

# In[179]:
apt_list = list(df.apt_date)

# In[180]:
empty = []
for i in range(len(list(df.apt_date))):
    index_val = apt_list[i].find(' ')
    empty.append(apt_list[i][:index_val])

# In[181]:
# fixing 
df.apt_date = empty

# In[182]:
#get a list of the categorical columns 
cat_list = (df.dtypes == 'object')

# In[183]:
# convert dates to date time format 
df.head()

# In[ ]:
# In[184]:
send_time_list = list(df.send_time)

# In[185]:
send_time_list[:5]


# In[186]:
empty_mins = []
empty_hours = []
for i in range(len(list(df.apt_date))):
    index_val = send_time_list[i].find(':')
    empty_mins.append(send_time_list[i][index_val+1:index_val+3])
    empty_hours.append(send_time_list[i][:index_val])


# In[187]:
empty_mins[:11]

# In[188]:
empty_hours[:10]

# In[189]:
df['send_time_mins'] = empty_mins
df['send_time_hours'] = empty_hours

# In[190]:
df.send_time_mins = df.send_time_mins.astype(int)

# In[191]:
df.send_time_hours = df.send_time_hours.astype(int)

# In[192]:
df.dtypes

# In[193]:
df.drop(['send_time'], axis=1, inplace=True)

# In[194]:
df.dtypes

# In[195]:
# creating a list of object headers
cat_list = (df.dtypes == 'object')

# In[196]:
# columns to pass to the get_dummies function 
col_names = list(df.dtypes[cat_list].index)

# - check number of values apt_type 
# - Send time to time format
# - We need to work on clinic_zip

# In[197]:
# Needs to be created as "Other"
list(df.apt_type.value_counts()[df.apt_type.value_counts()<3000].index)[0:5]

# In[198]:
# Our others list
others_list = list(df.apt_type.value_counts()[df.apt_type.value_counts()<3000].index)

# In[199]:
# empty dataframe creation 
df_others = pd.DataFrame(columns = df.columns)

# In[200]:
# Defining our function to create "others"
def f(row):
    val = row['apt_type']
    if row['apt_type'] in others_list:
        val = "others"
    return val

# In[201]:
# copying over the df dataframe
df_others = df.copy()

# In[202]:
# Applying the fucntion 
df_others['apt_type'] = df_others.apply(f, axis=1)


# In[203]:
df_others.apt_type.value_counts()[:5]

# In[56]:
# dont need to do anything for ages 
df_others.age.value_counts()

# In[204]:
zip_list = list(df_others.cli_zip)

# In[205]:
empty_3 = []
for i in range(len(zip_list)):
    empty_3.append(zip_list[i][:2])

# In[206]:
df_others.cli_zip = empty_3

# In[207]:
df_others.cli_zip[0:5]

# In[208]:
col_names = list(df.dtypes[cat_list].index)

# In[209]:
#use get_dummies function from pandas to convert categorical columns into numerical 
new_df = pd.get_dummies(df_others, columns = col_names).copy()

# In[63]:
list(new_df.columns)

# In[210]:
# No NULLS!!!
(new_df.isna().sum()>0).sum()

# In[211]:
# create our own bins 
new_df.head()

# In[162]:
list(new_df.columns)[1:]

# In[212]:
X = new_df[list(new_df.columns)[1:]]

# In[213]:
Y = new_df['response']

# In[214]:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# In[215]:
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# In[216]:
y_pred = clf.predict(X_test)

# In[217]:
accuracy = (y_pred == y_test).mean()

# In[218]:
accuracy 

# In[90]:
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:


"""
### Create df for test.csv
df = pd.read_csv('test.csv')

# get day of the week

empty_sent_dayofweek = []

sent_list = pd.to_datetime(df['sent_time'])

for i in range(len(list(df.apt_date))):
    empty_sent_dayofweek.append(sent_list[i].dayofweek)
    
df['sent_dayofweek'] = empty_sent_dayofweek

sent_dayofweek_list = list(df.sent_dayofweek)

empty_sent_weekend = []

for i in range(len(list(df.apt_date))):
    if sent_dayofweek_list[i] >= 5:
        empty_sent_weekend.append(1)
    else:
        empty_sent_weekend.append(0)
    
df['sent_weekend'] = empty_sent_weekend

df.drop(['clinic', 'sent_dayofweek', 'ReminderId', 'pat_id', 'family_id'], axis=1,inplace = True)

sent_list = list(df.sent_time)

empty_1 = []
for i in range(len(list(df.apt_date))):
    index_val = sent_list[i].find(' ')
    empty_1.append(sent_list[i][:index_val])

df.sent_time = empty_1

apt_list = list(df.apt_date)
empty = []
for i in range(len(list(df.apt_date))):
    index_val = apt_list[i].find(' ')
    empty.append(apt_list[i][:index_val])

df['send_time_mins'] = empty_mins
df['send_time_hours'] = empty_hours

df.send_time_mins = df.send_time_mins.astype(int)
df.send_time_hours = df.send_time_hours.astype(int)

df.drop(['send_time'], axis=1, inplace=True)


"""

