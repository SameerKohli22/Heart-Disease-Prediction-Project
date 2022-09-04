#!/usr/bin/env python
# coding: utf-8

# In[94]:


#IMPORTING THE DEPENDENCIES


# In[95]:


import numpy as np #for using arrays in python 


# In[96]:


from sklearn.model_selection import train_test_split #for splitting the training and testing data


# In[97]:


from sklearn.linear_model import LogisticRegression #for creating model of LogisticRegression algorithm 


# In[98]:


from sklearn.metrics import accuracy_score #for checking the accuracy of our model


# In[99]:


import pandas as pd #for working with files (eg . csv file in this case )


# In[100]:


#DATA COLLECTION AND PROCESSING


# In[101]:


#loading csv data to pandas dataframe
dataset=pd.read_csv('heart.csv')


# In[102]:


#Printing first 5 rows of dataset

dataset.head()


# In[103]:


#Printing last five rows of dataset
dataset.tail()


# In[104]:


#Number of rows and columns in dataset
dataset.shape


# In[105]:


#Getting some more information about dataset
dataset.info()


# In[106]:


#Checking for missing values

dataset.isnull().sum()


# In[107]:


#Statistical measures about the data

dataset.describe()


# In[108]:


#Checking the distribution of target variable

dataset['target'].value_counts() 

#Checking how many persons have heart disease and how many don't
# 0 - No heart disease(Healthy Heart)
# 1- Heart disease(Defective Heart)


# In[61]:


# Here distribution is almost even


# In[62]:


# Splitting the features and the target column

X = dataset.drop(columns='target',axis=1) #Separating the target column from X 

# X will only contain features now


# In[63]:


Y = dataset['target']

# Y will contain only the target column


# In[64]:


X

#Printing X (features columns without target column)


# In[65]:


Y

#Printing Y(target column)


# In[66]:


#Splitting data into training and test data

X_train,X_test,Y_train,Y_test = train_test_split(X , Y , test_size = 0.2, stratify = Y , random_state = 2)


# In[67]:


X.shape


# In[68]:


X_train.shape


# In[69]:


X_test.shape


# In[70]:


# MODEL TRAINING 


# In[71]:


# Using a logistic regression model as it is very good for binary classification 


# In[72]:


model = LogisticRegression()


# In[73]:


model


# In[74]:


# Training our LogisticRegression model with training values


model.fit(X_train,Y_train)

# this will find the pattern between the features contained in X_train and target values contained in Y_train


# In[75]:


# Model evaluation by accuracy score


# In[76]:


# accuracy on training data

X_train_prediction = model.predict(X_train)


# In[77]:


training_data_accuracy = accuracy_score(X_train_prediction , Y_train)


# In[78]:


training_data_accuracy


# In[79]:


# from the above prediction we get that our model out of 100 can predict 85 values correctly


# In[80]:


# accuracy score on testing data

X_test_prediction = model.predict(X_test)


# In[81]:


testing_data_accuracy = accuracy_score(X_test_prediction , Y_test)


# In[82]:


testing_data_accuracy


# In[83]:


# ACCURACY SCORE OF TRAINING DATA AND TESTING DATA ARE ALMOST SIMILAR


# In[84]:


# Now building a Predictive system


# In[85]:


input_data = (53,1,0,140,203,1,0,155,1,3.1,0,0,3) # tuple datatype


# In[86]:


# converting input_data to numpy array


# In[87]:


input_data_as_numpy_array = np.asarray(input_data)


# In[88]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the numpy array to tell our machine we are predicting for only one person


# In[89]:


prediction = model.predict(input_data_reshaped)


# In[90]:


if (prediction[0] == 0) :
    print("The person does not have a heart disease")
else:
    print("The person have a heart disease")


# In[92]:


import joblib


# In[93]:


joblib.dump(model,'prediction_model.pk1')


# In[ ]:




