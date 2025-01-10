#!/usr/bin/env python
# coding: utf-8

# # Black Friday Dataset EDA and Feature Engineering
# 
# 
# 
# ## Cleaning and Preparing the data for model training

# # Problem Statement
# 
# A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_train=pd.read_csv('train.csv')
df_train.head()


# In[3]:


df_test=pd.read_csv('test.csv')
df_test.head()


# In[4]:


df = pd.concat([df_train, df_test], ignore_index=True)
df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.drop(['User_ID'],axis=1,inplace=True)


# In[9]:


df.head()


# In[10]:


pd.get_dummies(df['Gender'])


# In[11]:


df['Gender']=df['Gender'].map({"F":0,'M':1})
df.head()


# ## Handle Categorical Feature Age

# In[12]:


df['Age'].unique()


# In[13]:


df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})
df.head()


# In[14]:


df_city=pd.get_dummies(df['City_Category'],drop_first=True)
df_city.head()


# In[15]:


df=pd.concat([df,df_city],axis=1)
df.head()


# In[16]:


df.drop('City_Category',axis=1,inplace=True)


# In[17]:


df.drop('Product_ID',axis=1,inplace=True)


# In[18]:


df.head()


# ## Missing values

# In[19]:


df.isnull().sum()


# Foucs on replacing missing values

# In[20]:


df['Product_Category_2'].unique()


# In[21]:


df['Product_Category_2'].value_counts()


# Replaceing the missing values with mode

# In[22]:


df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[23]:


df['Product_Category_2'].isnull().sum()


# In[24]:


df['Product_Category_3'].unique()


# In[25]:


df['Product_Category_3'].value_counts()


# In[26]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[27]:


df.head()


# In[28]:


df['Stay_In_Current_City_Years'].unique()


# In[29]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')


# In[30]:


df.head()


# In[31]:


df.info()


# #### convert object into intiger

# In[32]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)


# In[33]:


df.info()


# In[34]:


sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)
plt.show()


# ### Observation:
# Purchasing of men is high then women

# In[35]:


sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)
plt.show()


# In[36]:


sns.barplot(x='Product_Category_1',y='Purchase',hue='Gender',data=df)
plt.show()


# In[37]:


sns.barplot(x='Product_Category_2',y='Purchase',hue='Gender',data=df)
plt.show()


# In[38]:


sns.barplot(x='Product_Category_3',y='Purchase',hue='Gender',data=df)
plt.show()


# ### Observation:
# Product Category 1 is more purchased 

# ## Feature Scaling

# In[39]:


df_test=df[df['Purchase'].isnull()]


# In[40]:


df_train=df[~df['Purchase'].isnull()]


# In[41]:


X=df_train.drop('Purchase',axis=1)


# In[42]:


y=df_train['Purchase']


# In[43]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_tarin,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# In[44]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:




