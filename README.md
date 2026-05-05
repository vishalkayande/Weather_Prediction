# Weather Prediction App

A Django-based web application that provides real-time weather forecasts and uses machine learning to predict rain chances for any city!

## Features

1.  **Real-time Weather Data:** Fetch current weather from OpenWeatherMap API for any city worldwide
2.  **Rain Prediction Model:** RandomForestClassifier ML model trained on historical weather data to predict rain probability
3.  **Hourly Forecast:** Interactive hourly weather forecast for the next few hours
4.  **Comparison View:** Side-by-side comparison of ML prediction vs Global Industry Standard (OpenWeather)
5.  **Responsive Design:** Modern, beautiful UI that works on all devices

## Tech Stack

-   **Backend:** Django (Python Web Framework)
-   **Machine Learning:** Scikit-learn (RandomForestClassifier, RandomForestRegressor, LabelEncoder)
-   **Data Handling:** Pandas, NumPy
-   **Weather API:** OpenWeatherMap API
-   **Frontend:** HTML, CSS, JavaScript, Bootstrap Icons

## ML Model Details

### Rain Prediction Model (`rain_model.pkl`)
-   **Algorithm:** RandomForestClassifier
-   **Features Used (16 total):**
    -   MinTemp, MaxTemp
    -   WindGustDir, WindGustSpeed
    -   Humidity, Pressure, Temp
    -   CloudCover, Evaporation, Sunshine
    -   WindSpeed_9am, WindSpeed_3pm
    -   Humidity_9am, Humidity_3pm
    -   Pressure_9am, Pressure_3pm

## Installation

### Prerequisites
-   Python 3.10 or higher
-   pip package manager

### Step-by-Step Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vishalkayande/Weather_Prediction.git
    cd Rain-Weather-Predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv myenv
    
    # Windows
    myenv\Scripts\activate
    
    # Linux/Mac
    source myenv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Navigate to Django project directory:**
    ```bash
    cd weatherProject
    ```

5.  **Set up the database (optional):**
    ```bash
    python manage.py migrate
    ```

6.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```

7.  **Open in your browser:**
    Go to `http://localhost:8000` in your web browser!

## How to Use

1.  In the search bar at the top, type any city name (e.g., "London", "Mumbai", "New York")
2.  Click the search button
3.  View:
    -   Current weather conditions in the center panel
    -   ML-based rain prediction and hourly forecast on the left
    -   Global Industry Standard forecast on the right

## Project Structure

```
Rain-Weather-Predictor/
├── weatherProject/          # Main Django project
│   ├── forecast/            # Main application
│   │   ├── templates/       # HTML templates
│   │   ├── static/          # CSS, JS, images
│   │   └── views.py         # Main views and logic
│   ├── weatherProject/      # Project settings
│   ├── rain_model.pkl       # Trained rain prediction model
│   ├── temp_model.pkl       # Temperature model
│   ├── label_encoder.pkl    # Wind direction encoder
│   ├── weather.csv          # Training dataset 
│   └── manage.py            # Django management script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dataset

The model is trained on a weather dataset (`weather.csv`) from Kaggle with over 17,000 historical weather samples.

## API Key

This project uses OpenWeatherMap API. The API key is already configured in `forecast/views.py` for demonstration purposes. For production use, please obtain your own free API key from [OpenWeatherMap](https://openweathermap.org/api).

## License

This project is for educational and demonstration purposes.
