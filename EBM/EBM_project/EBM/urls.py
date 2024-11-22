from django.urls import path
from .views import PatientsListView, PatientVisitsView, PredictDiagnosisView, SavePatientVisitView

urlpatterns = [
    path('patients/', PatientsListView.as_view(), name='patients-list'),
    path('patients/<int:patient_id>/visits/', PatientVisitsView.as_view(), name='patient-visits'),
    path('predict/', PredictDiagnosisView.as_view(), name='predict-diagnosis'),
    path('save_visit/', SavePatientVisitView.as_view(), name='save-visit')
]

