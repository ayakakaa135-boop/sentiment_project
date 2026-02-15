from django.db import models
from django.utils import timezone

class SentimentPrediction(models.Model):
    """Store sentiment analysis predictions"""
    
    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
    ]
    
    text = models.TextField(help_text="Original text for analysis")
    cleaned_text = models.TextField(help_text="Preprocessed text")
    sentiment = models.CharField(
        max_length=10, 
        choices=SENTIMENT_CHOICES,
        help_text="Predicted sentiment"
    )
    confidence = models.FloatField(
        default=0.0,
        help_text="Prediction confidence score"
    )
    created_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Sentiment Prediction'
        verbose_name_plural = 'Sentiment Predictions'
    
    def __str__(self):
        return f"{self.sentiment} - {self.text[:50]}"

class AnalyticsStats(models.Model):
    """Store daily analytics statistics"""
    
    date = models.DateField(unique=True)
    total_predictions = models.IntegerField(default=0)
    positive_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-date']
        verbose_name = 'Analytics Stats'
        verbose_name_plural = 'Analytics Stats'
    
    def __str__(self):
        return f"Stats for {self.date}"
