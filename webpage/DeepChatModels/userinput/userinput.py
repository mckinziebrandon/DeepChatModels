from django import forms

class UserInputForm(forms.Form):
    text = forms.CharField(required=True, label="")
