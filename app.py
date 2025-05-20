import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encode_gender = pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    one_hot_enocder = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)

## Stream Lit App

st.title("Customer Churn Prediction")

##User Input 

geography = st.selectbox("Geography",one_hot_enocder.categories_[0])
gender = st.selectbox('Gender',label_encode_gender.classes_)
age = st.number_input("Age",min_value=18,max_value=100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",min_value=0,max_value=10)
num_of_products = st.slider("Number of Products",min_value=1,max_value=4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

input_data = {
    "CreditScore": [credit_score],
    "Gender" : [gender],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember" : [is_active_member],
    "EstimatedSalary" : [estimated_salary] 
}
input_data = pd.DataFrame(input_data)
geo_encoded = one_hot_enocder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=one_hot_enocder.get_feature_names_out())


input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data["Gender"] = label_encode_gender.transform(input_data["Gender"])
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")
st.write(f"Prediction Probability: {prediction_prob:.2f}")