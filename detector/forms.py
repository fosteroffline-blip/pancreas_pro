from django import forms

class ScanForm(forms.Form):
    patient_name = forms.CharField()
    file = forms.FileField()