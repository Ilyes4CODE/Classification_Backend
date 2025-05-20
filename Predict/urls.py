from django.urls import path
from .views import predict_disease, SendDiagnosisEmailView

urlpatterns = [
    path('predict/', predict_disease, name='predict_disease'),
    path('send-diagnosis-email/', SendDiagnosisEmailView.as_view(), name='send-diagnosis-email'),

]
