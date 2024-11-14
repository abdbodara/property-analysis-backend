from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    street_name = serializers.CharField(required=True)