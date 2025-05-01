# Must be FIRST Streamlit command
import streamlit as st
st.set_page_config(layout="wide", page_title="Climate Dashboard", page_icon="🌍")

# Import other libraries
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
import requests

# 🎨 Background Styling
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to bottom, rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.9)),
                        url('https://images.unsplash.com/photo-1592210454359-9043f067919b?q=80&w=2070&auto=format&fit=crop') no-repeat center center fixed;
            background-size: cover;
        }
        h1, h2, h3 {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load your dataset
df = pd.read_csv("GlobalWeatherRepository.csv")

# Sidebar navigation
st.sidebar.title("🌐 Navigate Dashboard")
section = st.sidebar.radio("Go to section:", [
    "Home",
    "Real-Time Weather",
    "Temperature Trends",
    "Air Quality Analysis",
    "Ozone Layer Stats",
    "Gas Emissions",
    "Glacier Stats",
    "Anomaly Detection",
    "Prediction Model"
])

# Weather API function
def fetch_real_time_data(city="New York"):
    api_key = "3accb8b3280e703a8391facdecd50774"
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

# 🏠 Home
if section == "Home":
    st.title("🌍 Climate Change & Global Warming Analysis Dashboard")
    st.write("Use the sidebar to explore real-time weather data, environmental trends, predictions, and more.")

# 🌦 Real-Time Weather
elif section == "Real-Time Weather":
    st.header("📡 Real-Time Weather")
    city = st.sidebar.text_input("Enter City", "New York")
    if st.sidebar.button("Fetch Weather"):
        data = fetch_real_time_data(city)
        if data:
            for k, v in data.items():
                st.write(f"**{k}:** {v}")

# 📈 Temperature Trends
elif section == "Temperature Trends":
    st.header("📈 Temperature Trends")
    fig_temp = px.line(df, x='last_updated', y='temperature_celsius', color='country')
    st.plotly_chart(fig_temp, use_container_width=True)

# 💨 Air Quality
elif section == "Air Quality Analysis":
    st.header("💨 Air Quality Analysis")
    fig_air = px.scatter(df, x='air_quality_Ozone', y='air_quality_PM2.5', color='country')
    st.plotly_chart(fig_air, use_container_width=True)

# 🌀 Ozone Stats
elif section == "Ozone Layer Stats":
    st.header("🌀 Ozone Layer Statistics (2000–2025)")
    ozone_data = pd.DataFrame({
        'Year': [2000, 2005, 2010, 2015, 2020, 2025],
        'Ozone Concentration (Dobson Units)': [290, 295, 300, 310, 320, 330]
    })
    fig_ozone = px.line(ozone_data, x='Year', y='Ozone Concentration (Dobson Units)', markers=True)
    st.plotly_chart(fig_ozone, use_container_width=True)

# 🌫️ Gas Emissions
elif section == "Gas Emissions":
    st.header("🌫️ Atmospheric Gas Composition & Emissions")
    gas_data = pd.DataFrame({
        'Gas': ['Nitrogen (N₂)', 'Oxygen (O₂)', 'Argon (Ar)', 'CO₂', 'Methane (CH₄)', 'Ozone (O₃)'],
        'Percentage': [78.08, 20.95, 0.93, 0.04, 0.0002, 0.000007]
    })
    st.plotly_chart(px.bar(gas_data, x='Gas', y='Percentage'), use_container_width=True)
    st.plotly_chart(px.pie(gas_data, names='Gas', values='Percentage'), use_container_width=True)

# 🧊 Glacier Stats
elif section == "Glacier Stats":
    st.header("🧊 Glacier Melt Trends & Projections")
    glacier_data = pd.DataFrame({
        'Year': [1990, 2000, 2010, 2020, 2025],
        'Global Glacier Volume Index': [100, 95, 87, 78, 70]
    })
    fig_glacier = px.area(glacier_data, x='Year', y='Global Glacier Volume Index')
    st.plotly_chart(fig_glacier, use_container_width=True)

# ⚠️ Anomaly Detection
elif section == "Anomaly Detection":
    st.header("⚠️ Anomaly Detection")
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'air_quality_PM2.5']])
    anomalies = df[df['anomaly'] == -1]
    st.write("Detected Anomalies:")
    st.dataframe(anomalies)
    fig_anom = px.scatter(df, x='temperature_celsius', y='air_quality_PM2.5', color=df['anomaly'].astype(str))
    st.plotly_chart(fig_anom, use_container_width=True)

# 🤖 Prediction Model
elif section == "Prediction Model":
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

    st.subheader("🔮 Predict Future PM2.5 Levels")
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

# 🔍 Learn More
with st.expander("📘 Learn More: Ozone & Atmosphere"):
    st.markdown("""
        - The **ozone layer** protects Earth from harmful UV rays.
        - **CO₂ and CH₄** are key contributors to global warming.
        - Monitoring **glacier melt** gives insight into rising sea levels.
    """)
