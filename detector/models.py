from django.db import models

class ScanResult(models.Model):
    patient_name = models.CharField(max_length=100)
    file = models.FileField(upload_to='scans/')
    result = models.CharField(max_length=200)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.patient_name