from django import forms

class SentimentForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Enter text to analyze sentiment...',
            'required': True
        }),
        label='Text to Analyze',
        max_length=5000
    )
