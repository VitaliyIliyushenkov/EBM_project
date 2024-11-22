from rest_framework import serializers
from .models import Patient, PatientVisit, Diagnosis

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'


class DiagnosisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Diagnosis
        fields = '__all__'


class PatientVisitSerializer(serializers.ModelSerializer):
    diagnosis = serializers.StringRelatedField()
    recommendations = serializers.CharField(source='diagnosis.recommendations', allow_null=True, required=False)  # Получение рекомендаций из связанного диагноза
    notes = serializers.CharField(allow_null=True, required=False)

    class Meta:
        model = PatientVisit
        fields = '__all__'

    def get_diagnosis_name(self, obj):
        if obj.diagnosis_id:  # Если указано поле diagnosis_id
            try:
                diagnosis = Diagnosis.objects.get(id=obj.diagnosis_id)
                return diagnosis.name
            except Diagnosis.DoesNotExist:
                return None
        return None
