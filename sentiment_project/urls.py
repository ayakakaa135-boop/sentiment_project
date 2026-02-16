# Main URLs configuration with i18n
# sentiment_project/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns

urlpatterns = [
    # Language switcher (without prefix)
    path('i18n/', include('django.conf.urls.i18n')),
]

# URLs with language prefix
urlpatterns += i18n_patterns(
    path('admin/', admin.site.urls),
    path('', include('sentiment_app.urls')),
    prefix_default_language=True,  # Set to False if you don't want /en/ prefix for English
)
