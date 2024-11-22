from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Patient, PatientVisit, Diagnosis
from .serializers import PatientSerializer, PatientVisitSerializer
from .casting import process_prediction, get_db_connection


class PatientsListView(APIView):
    """
    API для получения списка пациентов.
    """
    def get(self, request):
        patients = Patient.objects.all()
        serializer = PatientSerializer(patients, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = PatientSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()  # Сохраняем нового пациента в базе данных
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PatientVisitsView(APIView):
    """
    API для получения списка визитов пациента.
    """
    def get(self, request, patient_id):
        try:
            patient = Patient.objects.get(pk=patient_id)
        except Patient.DoesNotExist:
            return Response({"error": "Patient not found"}, status=status.HTTP_404_NOT_FOUND)

        visits = patient.visits.all().order_by('-visit_date')
        for visit in visits:
            if visit.diagnosis:
                visit.recommendations = visit.diagnosis.recommendations
            else:
                visit.recommendations = None

        serializer = PatientVisitSerializer(visits, many=True)
        return Response(serializer.data)


class PredictDiagnosisView(APIView):
    def post(self, request):
        # Получаем данные из запроса
        data = request.data
        try:
            # Вызываем функцию предсказания
            result = process_prediction(data)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SavePatientVisitView(APIView):
    def post(self, request):
        data = request.data
        try:
            # Получаем данные из запроса
            patient_id = data.get("patient_id")
            if not patient_id:
                full_name = data.get("full_name")
                birth_date = data.get("birth_date")
                # Проверяем существование пациента
                connection = get_db_connection()
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT id FROM patients WHERE full_name = %s AND birth_date = %s",
                        (full_name, birth_date)
                    )
                    patient = cursor.fetchone()
                    if patient:
                        patient_id = patient[0]
                    else:
                        cursor.execute(
                            "INSERT INTO patients (full_name, birth_date) VALUES (%s, %s) RETURNING id",
                            (full_name, birth_date)
                        )
                        patient_id = cursor.fetchone()[0]
                        connection.commit()

            # Сохраняем визит
            visit = PatientVisit(
                patient_id=patient_id,
                complaints=data.get("complaints", ""),
                disease_history=data.get("disease_history", ""),
                objective_status=data.get("objective_status", ""),
                age_category=data.get("age", ""),
                diagnosis_id=data.get("diagnosis_id"),
                notes=data.get("notes", ""),
                recommendations=", ".join(data.get("recommendations", []))
            )
            visit.save()
            return Response({"message": "Визит успешно сохранен."}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)