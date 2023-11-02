
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras.backend as K

# Define the root_mean_squared_error metric
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Load the LSTM model
model_path = '11epoch_iot_model.h5'
model = load_model(model_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})

# Load dataset (adjust path accordingly)
df = pd.read_excel("YOUR_PATH_TO_DATA/capped_hour_data_iot.xlsx")
cols = list(df)[1:7]
df_for_training = df[cols]

# Standard Scaler Initialization and transformation
scaler = StandardScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)

st.title("IoT Data Future Prediction with LSTM")

# User input for n_future
n_future = st.number_input("Enter the number of hours for future prediction:", min_value=1, max_value=720, step=1)

if st.button('Predict'):
    forecast = model.predict(df_for_training_scaled[-n_future:])
    forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(forecast_copies)

    # Extracting individual predictions
    actualVibx, predictedVibx = df_for_training["vibx"][-n_future:].values, y_pred_future[:, 0]
    actualVibz, predictedVibz = df_for_training["vibz"][-n_future:].values, y_pred_future[:, 1]
    actualTemp, predictedTemp = df_for_training["temp"][-n_future:].values, y_pred_future[:, 2]
    actualZacc, predictedZacc = df_for_training["zacc"][-n_future:].values, y_pred_future[:, 3]
    actualZfreq, predictedZfreq = df_for_training["zfreq"][-n_future:].values, y_pred_future[:, 4]
    actualXkurt, predictedXkurt = df_for_training["xkurt"][-n_future:].values, y_pred_future[:, 5]
    
    # Creating subplots
    fig, axs = plt.subplots(6, 1, figsize=(10, 20))
    
    # Plotting for each variable
    variables = [(actualVibx, predictedVibx, "VibX"), 
                 (actualVibz, predictedVibz, "VibZ"), 
                 (actualTemp, predictedTemp, "Temp"), 
                 (actualZacc, predictedZacc, "ZAcc"), 
                 (actualZfreq, predictedZfreq, "ZFreq"), 
                 (actualXkurt, predictedXkurt, "XKurt")]
    
    for i, (actual, predicted, label) in enumerate(variables):
        axs[i].plot(actual, label=f"Actual {label}", color="red")
        axs[i].plot(predicted, label=f"Predicted {label}", color="black")
        axs[i].set_title(f"{label} Forecast")
        axs[i].legend()

        # Display MAPE
        error = mae(actual, predicted)
        mean_value = df[label.lower()].mean()
        mape = (error / mean_value) * 100
        st.write(f"MAPE for {label}: {mape:.2f}%")

    st.pyplot(fig)

