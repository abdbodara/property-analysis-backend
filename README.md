#  Getting Started
#  Prerequisites
    Ensure the following are installed on your system:
        Python (version 3.7+)
        Django (version 3.2+)


# Property Price Prediction
This project predicts property prices based on tree height categories in the area. The data is processed, analyzed, and used to train a machine learning model in Python. A Django REST API serves predictions based on street names, and a React frontend (using TypeScript) displays the analysis insights.

# Project Structure
backend :- Django REST API
Machine Learning/Data Analysis

# Set up Project

# 1: Clone the Repository
    cd property_price

# 2: Set Up a Virtual Environment
    python3 -m venv venv
    source venv/bin/activate

# 3: Install Dependencies
    pip install -r requirements.txt

# 4: Set Up the Database and Django Server
    - python manage.py migrate
    - python manage.py runserver


#  API Usage

Endpoint: /api/predict-price/
Method: POST
Description: Predicts the property price based on a given street name.

# Request Body:-
{
  "street_name": "some_street_name"
}

# Response:-
{
  "predicted_price": 350000.00
}

# Example Request with curl
curl -X POST http://127.0.0.1:8000/api/predict-price/ -H "Content-Type: application/json" -d '{"street_name": "some_street_name"}'


#  Machine Learning Details
    1.Data Analysis: Data is loaded, processed, and merged, analyzing trends and correlations between property prices and tree height categories.
    2.Feature Engineering: Tree height is encoded for use as a model feature.
    3.Model Selection: A Random Forest Regressor is used for prediction, chosen for its robustness and effectiveness with categorical features.
    4.Model Evaluation: The model is evaluated using RMSE, MAE, and R², and predictions are validated with actual vs. predicted price visualizations.