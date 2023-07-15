#!/usr/bin/env python
# coding: utf-8

# # import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# 

# # Data Collection & Processing
# Load Data
# 

# In[7]:


# load the data from csv file

titanic_data=pd.read_csv("archieve.csv")


# In[8]:


# Print data

titanic_data.head()


# In[9]:


# Total number of rows & Columns
   
titanic_data.shape


# In[10]:


# Some informations about the data

titanic_data.info()


# In[11]:


# Missing value 

titanic_data.isnull().sum()


# # Handling the Missing values

# In[12]:


# drop cabin table 

titanic_data=titanic_data.drop(columns='Cabin',axis=1)


# In[13]:


#Replacing the missing values in "Age" column with mean value of age column

titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[14]:


# Search the mode value of "Embarked" column

print(titanic_data['Embarked'].mode())


# In[15]:


print(titanic_data['Embarked'].mode()[0])


# In[16]:


#Replacing the missing values in "Embarked" column with mode values

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[17]:


#Replacing the missing values in "Fare" column with mean values of fare column

titanic_data['Fare'].fillna(titanic_data['Fare'].mean(),inplace=True)


# In[18]:


#After filling missing values check again the number of missing values in each column

titanic_data.isnull().sum()


# # Data Analysis
# 
# 

# In[19]:


#Getting some statistical information about the data

titanic_data.describe()


# In[20]:


# Finding the number of survived and not survived people

titanic_data['Survived'].value_counts()


# # Data Visualization

# In[21]:


sns.set()


# In[22]:


#Making a count plot for "Survived" column

sns.countplot(x='Survived', data=titanic_data)


# In[23]:


titanic_data['Sex'].value_counts()


# In[24]:


#Making a count plot for "Sex" column

sns.countplot(x='Sex', data=titanic_data)


# In[25]:


#Number of survivors Gender wise

sns.countplot(x='Sex', hue='Survived',data=titanic_data)


# In[26]:


# making a count plot for "Pclass" column

sns.countplot(x='Pclass', data=titanic_data)


# In[27]:


sns.countplot(x='Pclass', hue='Survived', data=titanic_data)


# # Encoding the Categorical Columns

# In[28]:


titanic_data['Sex'].value_counts()


# In[29]:


titanic_data['Embarked'].value_counts()


# In[30]:


#Converting categorical Columns

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[31]:


titanic_data.head()


# # Separating features & Target

# In[32]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[33]:


print(X)


# In[34]:


print(Y)


# # Splitting the data into training data & Test data

# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[36]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training
# 
# Logistic Regression

# In[49]:


model = LogisticRegression(max_iter=1000)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)  # X is your input data

model = LogisticRegression()
model.fit(scaled_data, Y )# y is your target variable
model = LogisticRegression(solver='saga')



# In[46]:


#Training the Logistic Regression model with training data

model.fit(X_train, Y_train)


# # Model Evaluation
# 
# Accuracy Score

# In[39]:


# accuracy on training data

X_train_prediction = model.predict(X_train)


# In[40]:


print(X_train_prediction)


# In[41]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[42]:


# accuracy on test data

X_test_prediction = model.predict(X_test)


# In[43]:


print(X_test_prediction)


# In[44]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:





# In[ ]:




