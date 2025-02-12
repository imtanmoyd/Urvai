import ee  
ee.Authenticate()  
ee.Initialize()

# Load NASA Global Precipitation Measurement (GPM) dataset
gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V06") \
        .filterDate("2023-01-01", "2024-01-01") \
        .select("precipitationCal")

# Load temperature data
temperature = ee.ImageCollection("NOAA/CFSV2/FOR6H") \
              .filterDate("2023-01-01", "2024-01-01") \
              .select("Temperature_height_above_ground")

# Convert to Pandas DataFrame
import geemap

df = geemap.ee_to_pandas(gpm.mean(), properties=["precipitationCal"])
df_temp = geemap.ee_to_pandas(temperature.mean(), properties=["Temperature_height_above_ground"])

# Prepare Data for LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Merge precipitation and temperature data
df["temperature"] = df_temp["Temperature_height_above_ground"]
df = df.dropna()

# Normalize Data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Prepare Sequence Data
def create_sequences(data, seq_length=10):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(df_scaled)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Reshape for LSTM

#Build and Train the LSTM Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(y.shape[1])  # Predict precipitation & temperature
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.1)

# Climate Intervention Simulation
import gym
import numpy as np
from gym import spaces

class ClimateEnv(gym.Env):
    def __init__(self):
        super(ClimateEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,))
        self.action_space = spaces.Discrete(3)  # 0: No Action, 1: Cloud Seeding, 2: Heat Deflection

    def step(self, action):
        precipitation, temperature = self.state
        if action == 1:  # Cloud Seeding
            precipitation += np.random.uniform(0.1, 0.3)
        elif action == 2:  # Heat Deflection
            temperature -= np.random.uniform(0.5, 1.0)

        reward = -abs(precipitation - 0.5) - abs(temperature - 0.3)  # Stability Reward
        self.state = np.clip([precipitation, temperature], 0, 1)

        return np.array(self.state), reward, False, {}

    def reset(self):
        self.state = np.random.uniform(0, 1, size=(2,))
        return np.array(self.state)

env = ClimateEnv()

from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

from fastapi import FastAPI
app = FastAPI()
model = tf.keras.models.load_model("weather_lstm_model.h5")

@app.get("/predict")
def predict_weather():
    sample_input = np.random.rand(1, 10, 2)  # Example input
    prediction = model.predict(sample_input).tolist()
    return {"precipitation": prediction[0][0], "temperature": prediction[0][1]}

# Run API with: uvicorn script_name:app --reload
