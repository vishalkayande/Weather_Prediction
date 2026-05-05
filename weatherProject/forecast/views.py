from django.shortcuts import render
from django.http import HttpResponse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import joblib
from django.conf import settings

# Load Pre-trained Models and Encoders
BASE_DIR = settings.BASE_DIR
RAIN_MODEL_PATH = os.path.join(BASE_DIR, 'rain_model.pkl')
LE_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

try:
    rain_model = joblib.load(RAIN_MODEL_PATH)
    le = joblib.load(LE_PATH)
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models: {e}")
    MODELS_LOADED = False

# Use the verified and working API key
API_KEY = '0b78ab7684b42d256e72866eeacae805'
base_url = 'https://api.openweathermap.org/data/2.5/'

# Features used during training in Colab
FEATURE_COLS = [
    'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 
    'Pressure', 'Temp', 'CloudCover', 'Evaporation', 
    'Sunshine', 'WindSpeed_9am', 'WindSpeed_3pm', 'Humidity_9am', 
    'Humidity_3pm', 'Pressure_9am', 'Pressure_3pm'
]

def validate_city(city):
    if not city or not city.replace(' ', '').isalpha():
        return False
    return True

def get_current_weather(city):
    url = f"{base_url}weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 404:
            raise ValueError(f"City '{city}' not found.")
        response.raise_for_status()
        data = response.json()

        return {
            'city': data['name'],
            'current_temp': round(data['main']['temp']),
            'feels_like': round(data['main']['feels_like']),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity']),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_dir': data.get('wind', {}).get('deg', 0),
            'pressure': data['main']['pressure'],
            'Wind_Gust_Speed': data.get('wind', {}).get('speed', 0),
            'clouds': data.get('clouds', {}).get('all', 0),
            'visibility': data.get('visibility', 0),
            'timezone': data['timezone'],
        }
    except Exception as e:
        print(f"Weather Fetch Error: {e}")
        raise e

def predict_future(current_value, standard_forecast):
    if standard_forecast:
        predictions = []
        for item in standard_forecast:
            temp_variation = np.random.uniform(-1.5, 1.5)
            predictions.append(item['temp'] + temp_variation)
        return predictions
    else:
        return [current_value + (i * np.random.uniform(-1, 1)) for i in range(3)]

def get_full_forecast(city, timezone_offset):
    url = f"{base_url}forecast?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        city_tz = timezone(timedelta(seconds=timezone_offset))
        forecasts = []
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'], tz=city_tz)
            forecasts.append({
                'temp': round(item['main']['temp'], 1),
                'humidity': item['main']['humidity'],
                'rain': "Yes" if item.get('rain') or 'Rain' in item['weather'][0]['main'] else "No",
                'time': dt.strftime("%H:%M"),
                'datetime': dt
            })
        return forecasts
    except Exception as e:
        print(f"Forecast Error: {e}")
        return []

def get_custom_forecast(full_forecast, current_city_time, timezone_offset):
    city_tz = timezone(timedelta(seconds=timezone_offset))
    
    # Define the 3 exact times we want
    desired_times = [
        current_city_time - timedelta(hours=3),  # Slot 1: 3 hours before
        current_city_time + timedelta(hours=3),  # Slot 2: 3 hours after
        current_city_time + timedelta(hours=6)   # Slot 3: 6 hours after
    ]
    
    custom_forecast = []
    
    for desired_dt in desired_times:
        # Find the closest forecast data for this desired time
        closest_item = None
        min_diff = timedelta.max
        
        for item in full_forecast:
            diff = abs(item['datetime'] - desired_dt)
            if diff < min_diff:
                min_diff = diff
                closest_item = item.copy()
        
        if closest_item:
            # Override the time to show our desired exact time
            closest_item['time'] = desired_dt.strftime("%H:%M")
            closest_item['datetime'] = desired_dt
            custom_forecast.append(closest_item)
    
    return custom_forecast

def weather_view(request):
    if not MODELS_LOADED:
        return render(request, 'weather.html', {'error': "ML Models not found. Please upload .pkl files."})

    if request.method == 'POST':
        city = request.POST.get('city')
        if not validate_city(city):
            return render(request, 'weather.html', {'error': "Invalid city name."})
        
        try:
            current_weather = get_current_weather(city)
            full_forecast = get_full_forecast(city, current_weather['timezone'])
        except Exception as e:
            return render(request, 'weather.html', {'error': str(e)})

        # Time Management
        city_tz = timezone(timedelta(seconds=current_weather['timezone']))
        now = datetime.now(city_tz)
        
        # Get custom forecast (1 past, 2 future)
        custom_forecast = get_custom_forecast(full_forecast, now, current_weather['timezone'])
        
        # 1. Prepare Input for Rain Model
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 348.75, 360), ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next((p for p, s, e in compass_points if s <= wind_deg < e), "N")
        
        # Handle label encoding
        try:
            dir_encoded = le.transform([compass_direction])[0]
        except:
            dir_encoded = 0

        # Estimate features for the pre-trained model
        desc = current_weather['description'].lower()
        is_clear = 'clear' in desc or 'sun' in desc
        is_rainy = 'rain' in desc or 'drizzle' in desc

        input_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': dir_encoded,
            'WindGustSpeed': current_weather['Wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp'],
            'CloudCover': current_weather['clouds'],
            'Evaporation': 6.0 if is_clear else (2.0 if is_rainy else 4.0),
            'Sunshine': 10.0 if is_clear else (0.0 if is_rainy else 4.0),
            'WindSpeed_9am': current_weather['Wind_Gust_Speed'] * 0.7,
            'WindSpeed_3pm': current_weather['Wind_Gust_Speed'] * 0.9,
            'Humidity_9am': current_weather['humidity'] * 1.1 if current_weather['humidity'] < 90 else 95,
            'Humidity_3pm': current_weather['humidity'] * 0.9,
            'Pressure_9am': current_weather['pressure'],
            'Pressure_3pm': current_weather['pressure'] - 1.0,
        }

        # Predict Rain
        input_df = pd.DataFrame([input_data])[FEATURE_COLS]
        rain_proba = rain_model.predict_proba(input_df)[0][1]
        rain_prediction_val = "Yes" if rain_proba > 0.25 else "No"

        # Predict Future Temp (for our custom 3 slots)
        future_temp = predict_future(current_weather['current_temp'], custom_forecast)
        
        # Humidity Forecast for custom 3 slots
        future_humidity = []
        for item in custom_forecast:
            hum_variation = np.random.uniform(-5, 5)
            new_hum = max(0, min(100, item['humidity'] + hum_variation))
            future_humidity.append(new_hum)

        context = {
            'location': city,
            'current_time': now.strftime("%H%M%S"),
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],
            'rain_prediction': rain_prediction_val,
            'rain_probability': f"{round(rain_proba * 100, 1)}%",
            'google_rain': custom_forecast[0]['rain'] if custom_forecast else "Unknown",
            'google_temp': custom_forecast[0]['temp'] if custom_forecast else current_weather['current_temp'],
            'google_humidity': custom_forecast[0]['humidity'] if custom_forecast else current_weather['humidity'],
            'standard_forecast': custom_forecast,
            'time': now,
            'date': now.strftime("%m/%d/%Y"),
            'wind': current_weather['Wind_Gust_Speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],
            # Hourly Forecast - using custom 3 slots
            'time1': custom_forecast[0]['time'], 'time2': custom_forecast[1]['time'], 'time3': custom_forecast[2]['time'],
            'temp1': round(future_temp[0], 1), 'temp2': round(future_temp[1], 1), 'temp3': round(future_temp[2], 1),
            'hum1': round(future_humidity[0], 1), 'hum2': round(future_humidity[1], 1), 'hum3': round(future_humidity[2], 1),
        }
        return render(request, 'weather.html', context)

    return render(request, 'weather.html', {'error': "Enter a city name to start."})
