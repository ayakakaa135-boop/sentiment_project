from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
import joblib
import re
import os
from .models import SentimentPrediction, AnalyticsStats
from .forms import SentimentForm

# Global variables for model and vectorizer
model = None
vectorizer = None
MODEL_LOADED = False

def landing_view(request):
    """Professional landing page"""
    return render(request, 'landing.html')

def load_models():
    """Load ML models on first request"""
    global model, vectorizer, MODEL_LOADED
    
    if MODEL_LOADED:
        return True
    
    try:
        model_path = settings.ML_MODEL_PATH
        vectorizer_path = settings.VECTORIZER_PATH
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"❌ Model files not found")
            return False
        
        print(f"Loading models...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        MODEL_LOADED = True
        print("✅ Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def detect_language(text):
    """Detect if text is Arabic or English"""
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))
    
    if total_chars == 0:
        return 'unknown'
    
    arabic_ratio = len(arabic_chars) / total_chars
    return 'arabic' if arabic_ratio > 0.3 else 'english'

def translate_to_english(text):
    """
    Translate Arabic text to English using Google Translate (googletrans)
    Install: pip install googletrans==3.1.0a0
    """
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src='ar', dest='en')
        return result.text, True
    except ImportError:
        print("⚠️ googletrans not installed. Install it with: pip install googletrans==3.1.0a0")
        return text, False
    except Exception as e:
        print(f"Translation error: {e}")
        return text, False

def clean_text(text):
    """Preprocess text for prediction"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def home_view(request):
    """Home page with prediction form"""
    load_models()
    
    context = {
        'form': SentimentForm(),
        'model_loaded': MODEL_LOADED,
    }
    return render(request, 'home.html', context)

def predict_view(request):
    """Handle sentiment prediction with multi-language support"""
    if request.method != 'POST':
        return redirect('home')
    
    if not load_models():
        context = {
            'error': 'Models could not be loaded.',
            'form': SentimentForm()
        }
        return render(request, 'home.html', context)
    
    form = SentimentForm(request.POST)
    
    if not form.is_valid():
        context = {
            'error': 'Please enter valid text.',
            'form': form
        }
        return render(request, 'home.html', context)
    
    original_text = form.cleaned_data['text']
    
    try:
        # Detect language
        language = detect_language(original_text)
        translated_text = original_text
        was_translated = False
        
        # Translate if Arabic
        if language == 'arabic':
            translated_text, was_translated = translate_to_english(original_text)
            if not was_translated:
                # Fallback: try to process anyway
                print("⚠️ Translation failed, processing original text")
        
        # Clean text
        cleaned = clean_text(translated_text)
        
        if not cleaned:
            raise ValueError("Text is empty after cleaning")
        
        # Predict
        text_vec = vectorizer.transform([cleaned])
        prediction = model.predict(text_vec)[0]
        
        # Get confidence
        try:
            probabilities = model.predict_proba(text_vec)[0]
            confidence = float(max(probabilities)) * 100
        except:
            confidence = 0.0
        
        # Translate sentiment to Arabic if needed
        sentiment_ar = {
            'positive': 'إيجابي',
            'negative': 'سلبي',
            'neutral': 'محايد'
        }
        
        # Save to database
        try:
            ip = get_client_ip(request)
            SentimentPrediction.objects.create(
                text=original_text,
                cleaned_text=cleaned,
                sentiment=prediction,
                confidence=confidence,
                ip_address=ip
            )
            
            # Update daily stats
            today = timezone.now().date()
            stats, created = AnalyticsStats.objects.get_or_create(
                date=today,
                defaults={
                    'total_predictions': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0
                }
            )
            stats.total_predictions += 1
            if prediction == 'positive':
                stats.positive_count += 1
            elif prediction == 'negative':
                stats.negative_count += 1
            else:
                stats.neutral_count += 1
            stats.save()
        except Exception as e:
            print(f"Warning: Could not save to database: {e}")
        
        # Return result
        context = {
            'prediction': prediction,
            'prediction_ar': sentiment_ar.get(prediction, prediction),
            'confidence': confidence,
            'text': original_text,
            'language': language,
            'was_translated': was_translated,
            'translated_text': translated_text if was_translated else None,
            'form': SentimentForm()
        }
        return render(request, 'result.html', context)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        context = {
            'error': f'Prediction failed: {str(e)}',
            'form': SentimentForm()
        }
        return render(request, 'home.html', context)

def history_view(request):
    """Show prediction history"""
    try:
        predictions = SentimentPrediction.objects.all()[:50]
        
        total = SentimentPrediction.objects.count()
        positive = SentimentPrediction.objects.filter(sentiment='positive').count()
        negative = SentimentPrediction.objects.filter(sentiment='negative').count()
        neutral = SentimentPrediction.objects.filter(sentiment='neutral').count()
        
        context = {
            'predictions': predictions,
            'total': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
        }
    except Exception as e:
        print(f"History view error: {e}")
        context = {
            'predictions': [],
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
        }
    
    return render(request, 'history.html', context)

def analytics_view(request):
    """Show analytics dashboard"""
    try:
        seven_days_ago = timezone.now().date() - timedelta(days=7)
        daily_stats = AnalyticsStats.objects.filter(
            date__gte=seven_days_ago
        ).order_by('date')
        
        total_predictions = SentimentPrediction.objects.count()
        sentiment_counts = list(SentimentPrediction.objects.values('sentiment').annotate(
            count=Count('sentiment')
        ))
        
        dates = [stat.date.strftime('%Y-%m-%d') for stat in daily_stats]
        positive_data = [stat.positive_count for stat in daily_stats]
        negative_data = [stat.negative_count for stat in daily_stats]
        neutral_data = [stat.neutral_count for stat in daily_stats]
        
        context = {
            'total_predictions': total_predictions,
            'sentiment_counts': sentiment_counts,
            'dates': dates,
            'positive_data': positive_data,
            'negative_data': negative_data,
            'neutral_data': neutral_data,
        }
    except Exception as e:
        print(f"Analytics view error: {e}")
        context = {
            'total_predictions': 0,
            'sentiment_counts': [],
            'dates': [],
            'positive_data': [],
            'negative_data': [],
            'neutral_data': [],
        }
    
    return render(request, 'analytics.html', context)

@csrf_exempt
def api_predict(request):
    """API endpoint for predictions with multi-language support"""
    if request.method != 'POST':
        return JsonResponse({
            'error': 'Method not allowed',
            'success': False
        }, status=405)
    
    text = request.POST.get('text', '')
    
    if not text:
        return JsonResponse({
            'error': 'No text provided',
            'success': False
        }, status=400)
    
    if not load_models():
        return JsonResponse({
            'error': 'Models not loaded',
            'success': False
        }, status=500)
    
    try:
        # Detect and translate if needed
        language = detect_language(text)
        translated_text = text
        was_translated = False
        
        if language == 'arabic':
            translated_text, was_translated = translate_to_english(text)
        
        # Clean and predict
        cleaned = clean_text(translated_text)
        text_vec = vectorizer.transform([cleaned])
        prediction = model.predict(text_vec)[0]
        
        # Get confidence
        try:
            probabilities = model.predict_proba(text_vec)[0]
            confidence = float(max(probabilities)) * 100
            
            proba_dict = {}
            for i, class_name in enumerate(model.classes_):
                proba_dict[class_name] = float(probabilities[i])
        except:
            confidence = 0.0
            proba_dict = {}
        
        return JsonResponse({
            'success': True,
            'text': text,
            'language': language,
            'was_translated': was_translated,
            'translated_text': translated_text if was_translated else None,
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': proba_dict
        })
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'success': False
        }, status=500)
