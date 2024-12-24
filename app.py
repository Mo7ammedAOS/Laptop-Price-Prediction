import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

# Load training data for the encoder
X_train = pd.read_csv("traineddata.csv")  # Replace with your actual file name

# Load data for reference (optional)
data = pd.read_csv("traineddata.csv") 

st.title("Laptop Price Predictor")

# Get user inputs 
company = st.selectbox('Brand', data['Company'].unique())
type = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
os = st.selectbox('OS', data['OpSys'].unique())
weight = st.number_input('Weight of the laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', data['CPU_name'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
ssd_hdd = st.selectbox('SSD + HDD (in GB)', [0, 1128, 1256, 2256, 1512, 756, 2128, 2512, 2000])
gpu = st.selectbox('GPU(in GB)', data['GPU_brand'].unique())

if st.button('Predict Price'):
    # Preprocess inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])
    ppi = ((X_resolution**2) + (Y_resolution**2))**0.5 / screen_size 

    # Create query array (ensure consistent order)
    query = np.array([company, type, ram, weight, 
                      touchscreen, ips, ppi, cpu, hdd, ssd, ssd_hdd, gpu, os]).reshape(1, -1) 

    # Identify categorical columns 
    categorical_cols = [0, 1, 7, 11, 12]  # Replace with actual indices of categorical features

    # Create and fit the OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore') 
    encoder.fit(X_train[:, categorical_cols]) 

    # Transform the query using the fitted encoder
    query[:, categorical_cols] = encoder.transform(query[:, categorical_cols]).toarray()

    # Make prediction
    prediction = int(np.exp(rf.predict(query)[0]))

    # Display prediction
    st.title(f"Predicted price for this laptop could be between {prediction-50} AED to {prediction+50} AED)