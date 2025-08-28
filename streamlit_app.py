import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (this already includes preprocessor + model)
rf_model = joblib.load('best_diamond_price_model.joblib')

st.title('Diamond Price Prediction App')

st.write("""
Enter the characteristics of the diamond to predict its price.
""")

# Inputs
carat = st.number_input('Carat', min_value=0.1, max_value=10.0, value=0.5)
cut = st.selectbox('Cut', ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
color = st.selectbox('Color', ['G', 'E', 'F', 'H', 'D', 'I', 'J'])
clarity = st.selectbox('Clarity', ['SI1', 'VS2', 'SI2', 'VS1', 'VVS2', 'VVS1', 'I1', 'IF'])
depth = st.number_input('Depth', min_value=0.0, max_value=100.0, value=61.5)
table = st.number_input('Table', min_value=0.0, max_value=100.0, value=55.0)
x = st.number_input('X (length in mm)', min_value=0.0, max_value=20.0, value=3.95)
y = st.number_input('Y (width in mm)', min_value=0.0, max_value=20.0, value=3.98)
z = st.number_input('Z (depth in mm)', min_value=0.0, max_value=20.0, value=2.43)

if st.button('Predict Price'):
    input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]],
                              columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
    input_data['volume'] = input_data['x'] * input_data['y'] * input_data['z']

    predicted_price = rf_model.predict(input_data)
    st.success(f'Predicted Diamond Price: ${predicted_price[0]:.2f}')


