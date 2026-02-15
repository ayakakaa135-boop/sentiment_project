from django.contrib import admin
from .models import SentimentPrediction, AnalyticsStats

@admin.register(SentimentPrediction)
class SentimentPredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'sentiment', 'confidence', 'text_preview', 'created_at']
    list_filter = ['sentiment', 'created_at']
    search_fields = ['text', 'cleaned_text']
    readonly_fields = ['created_at']
    
    def text_preview(self, obj):
        return obj.text[:50] + '...' if len(obj.text) > 50 else obj.text
    text_preview.short_description = 'Text'

@admin.register(AnalyticsStats)
class AnalyticsStatsAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_predictions', 'positive_count', 'negative_count', 'neutral_count']
    list_filter = ['date']
    readonly_fields = ['date']
