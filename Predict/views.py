import joblib
import numpy as np
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

model = joblib.load('disease_prediction_model.pkl')
description_df = pd.read_csv('symptom_Description.csv')
precaution_df = pd.read_csv('symptom_precaution.csv')
description_dict = dict(zip(description_df['Disease'], description_df['Description']))
precaution_dict = dict(zip(
    precaution_df['Disease'], 
    precaution_df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
))

@api_view(['POST'])
def predict_disease(request):
    data = request.data.get('symptoms')

    if not isinstance(data, list):
        return Response({"error": "Symptoms must be a list of 17 numbers."}, status=status.HTTP_400_BAD_REQUEST)
    if len(data) != 17:
        return Response({"error": "Exactly 17 symptom weights are required."}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        features = np.array(data).reshape(1, -1)
        prediction = model.predict(features)[0]

        response_data = {
            "predicted_disease": prediction,
            "description": description_dict.get(prediction, "No description available."),
            "precautions": [p for p in precaution_dict.get(prediction, []) if p]
        }
        return Response(response_data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
