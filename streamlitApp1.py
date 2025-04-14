import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample bird flight data
data = {
    'Speed (km/h)': [10, 15, 20, 25, 30],
    'Wing Size (cm)': [15, 20, 25, 30, 35],
    'Distance (m)': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Train a simple model
X = df[['Speed (km/h)', 'Wing Size (cm)']]
y = df['Distance (m)']
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title("🕊️ Bird Flight Distance Predictor")

# Show the dataset
st.subheader("Training Data")
st.dataframe(df)

# User inputs
st.subheader("Enter Bird Info")
speed = st.slider("Speed (km/h)", 5, 50, 20)
wing_size = st.slider("Wing Size (cm)", 10, 50, 25)

# Make prediction
user_input = pd.DataFrame([[speed, wing_size]], columns=['Speed (km/h)', 'Wing Size (cm)'])
prediction = model.predict(user_input)[0]

# Display prediction
st.success(f"Estimated Flight Distance: {prediction:.2f} meters")
