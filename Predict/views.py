import joblib
import numpy as np
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.mail import EmailMessage
from django.conf import settings
import os
import logging

logger = logging.getLogger(__name__)
model = joblib.load('disease_prediction_model.pkl')
description_df = pd.read_csv('symptom_Description.csv')
precaution_df = pd.read_csv('symptom_precaution.csv')
description_dict = dict(zip(description_df['Disease'], description_df['Description']))
precaution_dict = dict(zip(
    precaution_df['Disease'], 
    precaution_df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
))

def remove_trailing_spaces(word):
    return word.rstrip()

@api_view(['POST'])
def predict_disease(request):
    data = request.data.get('symptoms')
    print("data = ",data)
    if not isinstance(data, list):
        return Response({"error": "Symptoms must be a list of 17 numbers."}, status=status.HTTP_400_BAD_REQUEST)
    if len(data) != 17:
        return Response({"error": "Exactly 17 symptom weights are required."}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        features = np.array(data).reshape(1, -1)
        print("features = ",features)
        print("teest: ",model.predict(features))
        prediction = model.predict(features)[0]
        prediction = remove_trailing_spaces(prediction)
        # print("description_dict = ",description_dict[prediction])
        response_data = {
            "predicted_disease": prediction,
            "description": description_dict.get(prediction, "No description available."),
            "precautions": [p for p in precaution_dict.get(prediction, []) if p]
        }
        return Response(response_data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    



class SendDiagnosisEmailView(APIView):
    """
    API endpoint that sends a diagnosis PDF via email
    """
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, format=None):
        try:
            # Extract data from request
            email = request.data.get('email')
            name = request.data.get('name', 'Patient')
            pdf_file = request.data.get('pdf_file')
            
            # Validate required fields
            if not email:
                return Response(
                    {'error': 'Email address is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            if not pdf_file:
                return Response(
                    {'error': 'PDF file is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Prepare email
            subject = "Your Smart Diagnosis Report"
            message = f"""
            Hello {name},
            
            Thank you for using Smart Diagnosis. Your diagnosis report is attached to this email.
            
            Please note that this is an automatically generated report from our system.
            The information contained in this report is for informational purposes only and 
            should not replace professional medical advice.
            
            Please consult with a healthcare professional regarding your diagnosis and treatment options.
            
            Best regards,
            The Smart Diagnosis Team
            """
            
            # Create and send email with attachment
            email_message = EmailMessage(
                subject=subject,
                body=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email]
            )
            
            # Attach the PDF file
            email_message.attach(
                pdf_file.name,
                pdf_file.read(),
                'application/pdf'
            )
            
            email_message.send(fail_silently=False)
            
            # Return success response
            return Response(
                {'success': 'Diagnosis report sent successfully'}, 
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            logger.error(f"Error sending diagnosis email: {str(e)}")
            return Response(
                {'error': 'Failed to send email. Please try again later.'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
