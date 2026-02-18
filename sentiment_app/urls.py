from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('history/', views.history_view, name='history'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/v2/predict/', views.api_predict_v2, name='api_predict_v2'),
    path('landing/', views.landing_view, name='landing'),
]
