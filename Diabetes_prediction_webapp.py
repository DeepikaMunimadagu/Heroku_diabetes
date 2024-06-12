import streamlit as st
import numpy as np
import pickle

# Function to load the model
@st.cache
def load_model(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

# Function to make predictions
def make_prediction(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

def main():
    # Load the model
    loaded_model = load_model(r'C:\Users\Dell\Desktop\ML\trained_model.sav')
    
    # Set up the layout of the app
    st.title('Diabetes Prediction Web App')
    
    # Input fields
    pregnancies = st.text_input('Number of Pregnancies:')
    glucose = st.text_input('Glucose Level:')
    blood_pressure = st.text_input('Blood Pressure:')
    skinthicknessvalue = st.text_input('Skin Thickness Value:')
    insulin = st.text_input('Insulin Level:')
    bmi = st.text_input('BMI:')
    diapedigreefun = st.text_input('Diabetes Pedigree Function Value:')
    age = st.text_input("Age of the person:")
    # Add more input fields for other features...
    
    # Prediction button
    if st.button('Diabetes Test Result'):
        input_data = [pregnancies, glucose, blood_pressure, skinthicknessvalue, insulin, bmi, diapedigreefun, age]  # Add more input data as needed
        try:
            # Make prediction
            prediction = make_prediction(loaded_model, input_data)
            if prediction[0] == 0:
                st.success('The person is not diabetic')
            else:
                st.success('The person is diabetic')
        except ValueError:
            st.error('Please enter valid numeric values for all input fields.')

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    