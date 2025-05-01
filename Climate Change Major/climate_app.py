import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
import requests

# Set up Streamlit page layout
st.set_page_config(layout="wide")

# Custom styling
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to bottom, rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.9)),
                        url('https://images.unsplash.com/photo-1592210454359-9043f067919b?q=80&w=2070&auto=format&fit=crop') no-repeat center center fixed;
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load climate dataset
file_path="GlobalWeatherRepository.csv"  # Ensure this file is present
df = pd.read_csv(file_path)

# Real-time weather API
def fetch_real_time_data(city="New York"):
    api_key = "3accb8b3280e703a8391facdecd50774"  # Replace with your key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'City': city,
                'Temperature (°C)': data['main']['temp'],
                'Humidity (%)': data['main']['humidity'],
                'Pressure (mb)': data['main']['pressure'],
                'Wind Speed (m/s)': data['wind']['speed'],
                'Weather Description': data['weather'][0]['description'],
            }
        else:
            st.sidebar.error(f"Error {response.status_code}: Unable to fetch data.")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API Error: {e}")
    return None

# Sidebar input for real-time weather
st.sidebar.header("Real-Time Weather Update")
city = st.sidebar.text_input("Enter City", "New York")
if st.sidebar.button("Fetch Weather"):
    data = fetch_real_time_data(city)
    if data:
        for k, v in data.items():
            st.sidebar.write(f"**{k}:** {v}")

# Main dashboard title
st.title("🌍 Climate Change & Global Warming Analysis Dashboard")

# Temperature trend visualization
st.header("📈 Temperature Trends")
fig_temp = px.line(df, x='last_updated', y='temperature_celsius', color='country')
st.plotly_chart(fig_temp, use_container_width=True)

# Air quality scatter plot
st.header("💨 Air Quality Analysis")
fig_air = px.scatter(df, x='air_quality_Ozone', y='air_quality_PM2.5', color='country')
st.plotly_chart(fig_air, use_container_width=True)

# Anomaly detection
st.header("⚠️ Anomaly Detection")
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'air_quality_PM2.5']])
anomalies = df[df['anomaly'] == -1]
st.write("Detected Anomalies:")
st.dataframe(anomalies)
fig_anom = px.scatter(df, x='temperature_celsius', y='air_quality_PM2.5', color=df['anomaly'].astype(str))
st.plotly_chart(fig_anom, use_container_width=True)

# Machine learning prediction model
st.header("🤖 Air Pollution Prediction (PM2.5)")
features = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph']
target = 'air_quality_PM2.5'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"📊 Model Mean Squared Error: {mse:.2f}")

# User input prediction
st.header("🔮 Predict Future PM2.5 Levels")
user_input = {
    'temperature_celsius': st.slider("Temperature (°C)", float(df['temperature_celsius'].min()), float(df['temperature_celsius'].max())),
    'humidity': st.slider("Humidity (%)", int(df['humidity'].min()), int(df['humidity'].max())),
    'pressure_mb': st.slider("Pressure (mb)", float(df['pressure_mb'].min()), float(df['pressure_mb'].max())),
    'wind_kph': st.slider("Wind Speed (kph)", float(df['wind_kph'].min()), float(df['wind_kph'].max()))
}

if st.button("Predict PM2.5 Level"):
    input_df = pd.DataFrame([user_input])
    predicted = rf_model.predict(input_df)[0]
    st.success(f"Predicted PM2.5 Level: {predicted:.2f}")
