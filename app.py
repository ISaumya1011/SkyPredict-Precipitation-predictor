from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import requests

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the trained model, scaler, and label encoders
model = joblib.load('xgboost_rain_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
le_encoders = joblib.load('label_encoders.pkl')
le_target = joblib.load('label_encoder_target.pkl')

# News API configuration
NEWS_API_URL = "https://newsapi.org/v2/everything"
API_KEY = "b5d8da685be540cf8aa20c9133ecf8ab"  # Your actual API key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/news')
def news():
    query = request.args.get('q', 'weather')  # Default query is 'weather'
    page = request.args.get('pageno', 1, type=int)  # Default page number is 1

    # Fetch articles from the API
    params = {
        'q': query,
        'apiKey': API_KEY,
        'page': page,
        'pageSize': 12  # Number of articles per page
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        articles = response.json().get('articles', [])
        total_results = response.json().get('totalResults', 0)
    except requests.exceptions.HTTPError as err:
        articles = []
        total_results = 0

    # Calculate total pages
    total_pages = (total_results // 12) + (1 if total_results % 12 > 0 else 0)

    return render_template('news.html', query=query, page=page, articles=articles, total_pages=total_pages)

@app.route('/api', methods=['GET'])
def api():
    query = request.args.get('q', 'weather')  # Default query is 'weather'
    page = request.args.get('pageno', 1)  # Default page number is 1
    params = {
        'q': query,
        'apiKey': API_KEY,
        'page': page,
        'pageSize': 12  # Number of articles per page
    }
    
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return jsonify(response.json())
    except requests.exceptions.HTTPError as err:
        return jsonify({"error": str(err)}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if POST request
    if request.method == 'POST':
        print("Received POST request")
        # Get form data
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rainfall = float(request.form['rainfall'])
        wind_gust_dir = request.form['wind_gust_dir']
        wind_gust_speed = float(request.form['wind_gust_speed'])
        wind_dir_9am = request.form['wind_dir_9am']
        wind_dir_3pm = request.form['wind_dir_3pm']
        wind_speed_9am = float(request.form['wind_speed_9am'])
        wind_speed_3pm = float(request.form['wind_speed_3pm'])
        humidity_9am = float(request.form['humidity_9am'])
        humidity_3pm = float(request.form['humidity_3pm'])
        pressure_9am = float(request.form['pressure_9am'])
        pressure_3pm = float(request.form['pressure_3pm'])
        cloud_9am = float(request.form['cloud_9am'])
        cloud_3pm = float(request.form['cloud_3pm'])
        temp_9am = float(request.form['temp_9am'])
        temp_3pm = float(request.form['temp_3pm'])

        # Encode categorical variables
        wind_gust_dir_encoded = le_encoders[3].transform([wind_gust_dir])[0]
        wind_dir_9am_encoded = le_encoders[5].transform([wind_dir_9am])[0]
        wind_dir_3pm_encoded = le_encoders[6].transform([wind_dir_3pm])[0]

        # Create feature array
        features = np.array([[min_temp, max_temp, rainfall , wind_gust_dir_encoded, wind_gust_speed,
                              wind_dir_9am_encoded, wind_dir_3pm_encoded, wind_speed_9am, wind_speed_3pm,
                              humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
                              cloud_9am, cloud_3pm, temp_9am, temp_3pm]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict with XGBoost model
        prediction = model.predict(features_scaled)

        # Interpret prediction
        result = "It may rain tomorrow." if prediction[0] == 1 else "It may not rain tomorrow."

        return result

if __name__ == '__main__':
    app.run(debug=True)