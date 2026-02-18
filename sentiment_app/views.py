from datetime import timedelta
import os
import re

import joblib
from django.conf import settings
from django.core.cache import cache
from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.cache import cache_page
from django.views.decorators.csrf import csrf_exempt
from .forms import SentimentForm
from .models import AnalyticsStats, SentimentPrediction

# Global variables for models
model = None
vectorizer = None
transformer_pipeline = None
MODEL_LOADED = False


def landing_view(request):
    """Professional landing page"""
    return render(request, 'landing.html')


def load_models():
    """Load ML models once. Optionally use HuggingFace Transformers."""
    global model, vectorizer, transformer_pipeline, MODEL_LOADED

    if MODEL_LOADED:
        return True

    use_transformers = os.getenv('USE_TRANSFORMERS', 'False') == 'True'
    if use_transformers:
        try:
            from transformers import pipeline

            model_name = os.getenv(
                'HF_MODEL_NAME',
                'cardiffnlp/twitter-roberta-base-sentiment-latest',
            )
            transformer_pipeline = pipeline('sentiment-analysis', model=model_name)
            MODEL_LOADED = True
            print(f"✅ Transformers model loaded: {model_name}")
            return True
        except Exception as exc:
            print(f"⚠️ Could not load transformers model, falling back to sklearn: {exc}")

    try:
        model_path = settings.ML_MODEL_PATH
        vectorizer_path = settings.VECTORIZER_PATH

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print("❌ Model files not found")
            return False

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        MODEL_LOADED = True
        print('✅ sklearn models loaded successfully!')
        return True
    except Exception as exc:
        print(f"❌ Error loading models: {exc}")
        return False


def detect_language(text):
    """Detect if text is Arabic or English."""
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))

    if total_chars == 0:
        return 'unknown'

    arabic_ratio = len(arabic_chars) / total_chars
    return 'arabic' if arabic_ratio > 0.3 else 'english'


def translate_to_english(text):
    """Translate Arabic text to English using googletrans."""
    try:
        from googletrans import Translator

        translator = Translator()
        result = translator.translate(text, src='ar', dest='en')
        return result.text, True
    except ImportError:
        print('⚠️ googletrans not installed. Install it with: pip install googletrans==3.1.0a0')
        return text, False
    except Exception as exc:
        print(f'Translation error: {exc}')
        return text, False


def clean_text(text):
    """Preprocess text for prediction. Uses NLTK if available."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())

    try:
        from nltk.stem import PorterStemmer
        from nltk.tokenize import RegexpTokenizer

        tokenizer = RegexpTokenizer(r'\w+')
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(tok) for tok in tokenizer.tokenize(text)]
        return ' '.join(tokens)
    except Exception:
        return text


def normalize_sentiment_label(label):
    """Normalize external labels to app labels."""
    label = str(label).lower()
    mapping = {
        'label_0': 'negative',
        'label_1': 'neutral',
        'label_2': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'positive': 'positive',
    }
    return mapping.get(label, label)


def get_prediction_payload(text):
    """Shared prediction workflow for web and API views."""
    language = detect_language(text)
    translated_text = text
    was_translated = False

    if language == 'arabic':
        translated_text, was_translated = translate_to_english(text)

    cleaned = clean_text(translated_text)
    if not cleaned:
        raise ValueError('Text is empty after cleaning')

    if transformer_pipeline is not None:
        results = transformer_pipeline(translated_text)
        top_result = results[0]
        prediction = normalize_sentiment_label(top_result.get('label', 'neutral'))
        confidence = float(top_result.get('score', 0.0)) * 100
        proba_dict = {normalize_sentiment_label(top_result.get('label', 'neutral')): float(top_result.get('score', 0.0))}
    else:
        text_vec = vectorizer.transform([cleaned])
        prediction = model.predict(text_vec)[0]
        try:
            probabilities = model.predict_proba(text_vec)[0]
            confidence = float(max(probabilities)) * 100
            proba_dict = {
                class_name: float(probabilities[i])
                for i, class_name in enumerate(model.classes_)
            }
        except Exception:
            confidence = 0.0
            proba_dict = {}

    return {
        'text': text,
        'language': language,
        'was_translated': was_translated,
        'translated_text': translated_text if was_translated else None,
        'cleaned_text': cleaned,
        'sentiment': prediction,
        'confidence': confidence,
        'probabilities': proba_dict,
    }


def get_client_ip(request):
    """Get client IP address."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0]
    return request.META.get('REMOTE_ADDR')


def persist_prediction(request, payload):
    """Persist prediction and update analytics counters."""
    ip = get_client_ip(request)
    SentimentPrediction.objects.create(
        text=payload['text'],
        cleaned_text=payload['cleaned_text'],
        sentiment=payload['sentiment'],
        confidence=payload['confidence'],
        ip_address=ip,
    )

    today = timezone.now().date()
    stats, _ = AnalyticsStats.objects.get_or_create(
        date=today,
        defaults={
            'total_predictions': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
        },
    )
    stats.total_predictions += 1
    if payload['sentiment'] == 'positive':
        stats.positive_count += 1
    elif payload['sentiment'] == 'negative':
        stats.negative_count += 1
    else:
        stats.neutral_count += 1
    stats.save()

    cache.delete('analytics_dashboard_v1')


def home_view(request):
    """Home page with prediction form."""
    load_models()
    context = {'form': SentimentForm(), 'model_loaded': MODEL_LOADED}
    return render(request, 'home.html', context)


def predict_view(request):
    """Handle sentiment prediction with multi-language support."""
    if request.method != 'POST':
        return redirect('home')

    if not load_models():
        return render(
            request,
            'home.html',
            {'error': 'Models could not be loaded.', 'form': SentimentForm()},
        )

    form = SentimentForm(request.POST)
    if not form.is_valid():
        return render(
            request,
            'home.html',
            {'error': 'Please enter valid text.', 'form': form},
        )

    original_text = form.cleaned_data['text']

    try:
        payload = get_prediction_payload(original_text)

        try:
            persist_prediction(request, payload)
        except Exception as exc:
            print(f'Warning: Could not save to database: {exc}')

        sentiment_ar = {'positive': 'إيجابي', 'negative': 'سلبي', 'neutral': 'محايد'}
        context = {
            'prediction': payload['sentiment'],
            'prediction_ar': sentiment_ar.get(payload['sentiment'], payload['sentiment']),
            'confidence': payload['confidence'],
            'text': original_text,
            'language': payload['language'],
            'was_translated': payload['was_translated'],
            'translated_text': payload['translated_text'],
            'form': SentimentForm(),
        }
        return render(request, 'result.html', context)
    except Exception as exc:
        print(f'Prediction error: {exc}')
        return render(
            request,
            'home.html',
            {'error': f'Prediction failed: {str(exc)}', 'form': SentimentForm()},
        )


def history_view(request):
    """Show prediction history."""
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
    except Exception as exc:
        print(f'History view error: {exc}')
        context = {
            'predictions': [],
            'total': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
        }

    return render(request, 'history.html', context)


@cache_page(60 * 2)
def analytics_view(request):
    """Show analytics dashboard with short-lived caching."""
    cache_key = 'analytics_dashboard_v1'
    context = cache.get(cache_key)
    if context:
        return render(request, 'analytics.html', context)

    try:
        seven_days_ago = timezone.now().date() - timedelta(days=7)
        daily_stats = AnalyticsStats.objects.filter(date__gte=seven_days_ago).order_by('date')

        total_predictions = SentimentPrediction.objects.count()
        sentiment_counts = list(
            SentimentPrediction.objects.values('sentiment').annotate(count=Count('sentiment'))
        )

        context = {
            'total_predictions': total_predictions,
            'sentiment_counts': sentiment_counts,
            'dates': [stat.date.strftime('%Y-%m-%d') for stat in daily_stats],
            'positive_data': [stat.positive_count for stat in daily_stats],
            'negative_data': [stat.negative_count for stat in daily_stats],
            'neutral_data': [stat.neutral_count for stat in daily_stats],
        }
    except Exception as exc:
        print(f'Analytics view error: {exc}')
        context = {
            'total_predictions': 0,
            'sentiment_counts': [],
            'dates': [],
            'positive_data': [],
            'negative_data': [],
            'neutral_data': [],
        }

    cache.set(cache_key, context, timeout=120)
    return render(request, 'analytics.html', context)


@csrf_exempt
def api_predict(request):
    """Legacy API endpoint for predictions with multi-language support."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed', 'success': False}, status=405)

    text = request.POST.get('text', '')
    if not text:
        return JsonResponse({'error': 'No text provided', 'success': False}, status=400)

    if not load_models():
        return JsonResponse({'error': 'Models not loaded', 'success': False}, status=500)

    try:
        payload = get_prediction_payload(text)
        return JsonResponse({'success': True, **payload})
    except Exception as exc:
        return JsonResponse({'error': str(exc), 'success': False}, status=500)




@csrf_exempt
def api_predict_v2(request):
    """JSON-first API endpoint for mobile/web clients."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed', 'success': False}, status=405)

    import json
    try:
        body = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        body = {}

    text = body.get('text') or request.POST.get('text', '')
    if not text:
        return JsonResponse({'error': 'No text provided', 'success': False}, status=400)

    if not load_models():
        return JsonResponse({'error': 'Models not loaded', 'success': False}, status=500)

    try:
        payload = get_prediction_payload(text)
        return JsonResponse({'success': True, **payload}, status=200)
    except Exception as exc:
        return JsonResponse({'error': str(exc), 'success': False}, status=500)
