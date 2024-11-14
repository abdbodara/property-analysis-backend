import json, io, base64
import pandas as pd
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from .serializers import PredictionInputSerializer
import matplotlib.pyplot as plt
import seaborn as sns
from django.http import HttpResponse


with open('./city-trees.json', 'r') as f:
    city_trees_data = json.load(f)

tree_data_list = []
for street, tree_info in city_trees_data.items():
    for location, height in tree_info.items():
        for sub_location, height_value in height.items():
            tree_data_list.append({
                'street_name': sub_location,
                'height_category': height_value
            })

tree_data = pd.DataFrame(tree_data_list)
property_data = pd.read_csv('./property-data.csv', encoding='latin1')
property_data['Price'] = property_data['Price'].replace(r'[£$,ï¿½]', '', regex=True).astype(float)
merged_data = pd.merge(property_data, tree_data, on='street_name', how='left')
encoder = LabelEncoder()
merged_data['height_category_encoded'] = encoder.fit_transform(merged_data['height_category'])

X = merged_data[['height_category_encoded']]
y = merged_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Define API view
class PricePredictionView(APIView):
    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)
        print('serializer: ', serializer)
        if serializer.is_valid():
            street_name = serializer.validated_data['street_name']
            height_category = merged_data[merged_data['street_name'] == street_name]['height_category_encoded'].values[0]            
            price_prediction = model.predict([[height_category]])
            print('price_prediction: ', price_prediction)

            return Response({'predicted_price': round(price_prediction[0],2)}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class DataSummaryView(APIView):
    def get(self, request):
        summary = merged_data.groupby('height_category')['Price'].agg(['mean', 'median', 'min', 'max']).to_dict()
        return Response({"summary": summary}, status=status.HTTP_200_OK)
    
class PriceDistributionView(APIView):
    def get(self, request):
        distribution = merged_data.groupby('height_category')['Price'].apply(list).to_dict()
        return Response(distribution, status=status.HTTP_200_OK)

class ModelEvaluationView(APIView):
    def get(self, request):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "Root Mean Square Error": rmse,
            "Mean Absolute Error": mae,
            "Mean Sqaure Error": mse,
            "r2_score": r2
        }
        return Response({"metrics": metrics}, status=status.HTTP_200_OK)

class PlotPriceByHeightView(APIView):
    def get(self, request):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=merged_data, x='height_category', y='Price')
        plt.title("Property Prices by Tree Height Categories")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        return HttpResponse(buffer, content_type='image/png')
    
class PlotActualVsPredictedView(APIView):
    def get(self, request):
        y_pred = model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs Predicted Property Prices")

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        return HttpResponse(buffer, content_type='image/png')
