#!/usr/bin/env python
# coding: utf-8

# In[1]:

pip install scikit-learn==<version> numpy==<version> streamlit==<version>


import sklearn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import time  # Import time module

# Install required libraries
get_ipython().system('pip install streamlit pandas numpy scikit-learn')



california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame

# Create a Streamlit UI
st.title('California Housing Predictor')

# User input
user_input_neighbors = st.sidebar.slider('Enter the number of neighbors', 1, 10, 3)
user_input_population = st.sidebar.slider('Enter the Population', 0, int(df["Population"].max()), 1000)
user_input_bedrooms = st.sidebar.slider('Enter the Average Bedrooms', 0, int(df["AveBedrms"].max()), 3)
user_input_occupancy = st.sidebar.slider('Enter the Average Occupancy', 0, int(df["AveOccup"].max()), 3)
user_input_rooms = st.sidebar.slider('Enter the Average Rooms', 0, int(df["AveRooms"].max()), 5)
user_input_age = st.sidebar.slider('Enter the House Age', 0, int(df["HouseAge"].max()), 20)
user_input_income = st.sidebar.slider('Enter the Median Income', 0, int(df["MedInc"].max()), 4)

# Split the data
X = df.drop(columns=['MedHouseVal', 'Latitude', 'Longitude'])  # Exclude Latitude and Longitude
Y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
regressor = KNeighborsRegressor(n_neighbors=user_input_neighbors)
regressor.fit(X_train_scaled, y_train)

# Create a dataframe for the user input
user_input_df = pd.DataFrame({
    'Population': [user_input_population],
    'AveBedrms': [user_input_bedrooms],
    'AveOccup': [user_input_occupancy],
    'AveRooms': [user_input_rooms],
    'HouseAge': [user_input_age],
    'MedInc': [user_input_income]
}, columns=X.columns)  # Ensure columns match training data

# Scale the user input
user_input_scaled = scaler.transform(user_input_df)

# Display a loading spinner while the prediction is being calculated
with st.spinner('Calculating...'):
    # Simulate a delay (remove this in real app)
    time.sleep(2)
    
    # Predict the median house value
    predicted_med_house_val = regressor.predict(user_input_scaled)

# Print the predicted median house value
st.write(f'The predicted median house value for the given inputs is: {predicted_med_house_val[0]}')

# Print the longitude and latitude
st.write(f'The longitude and latitude are: {df["Longitude"].iloc[0]}, {df["Latitude"].iloc[0]}')




# In[2]:




