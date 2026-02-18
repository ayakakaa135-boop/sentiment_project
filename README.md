# 🤖 Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2-green.svg)](https://djangoproject.com)
[![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-84%25-success.svg)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An advanced, production-ready sentiment analysis platform powered by machine learning. Supports English and Arabic with automatic language detection and translation.

![Platform Screenshot](https://via.placeholder.com/1200x600/667eea/ffffff?text=Sentiment+Analysis+Platform)

---

## 🌟 Key Features

### 🎯 Core Functionality
- **High-Accuracy ML Model**: Random Forest classifier with 84.45% accuracy
- **Multi-Language Support**: English & Arabic with automatic detection
- **Real-Time Analysis**: Instant sentiment classification
- **Confidence Scoring**: Probability distribution for each prediction

### 📊 Analytics & Insights
- **Interactive Dashboards**: Multiple chart types (Pie, Doughnut, Line)
- **Historical Data**: Complete prediction history with filtering
- **Trend Analysis**: 7-day sentiment trends
- **Export Capabilities**: Download data as CSV

### 🔌 Integration
- **REST API**: Full JSON API + DRF v2 endpoint for external integrations
- **Multilingual**: Built-in translation engine
- **Scalable**: Optimized for high-volume processing

---

### ⚡ Performance & Accuracy Enhancements
- **Redis Caching (Optional)**: Cache analytics responses for faster repeated requests.
- **Transformers Backend (Optional)**: Set `USE_TRANSFORMERS=True` to use a HuggingFace model.
- **NLTK Preprocessing**: Improved tokenization and stemming before prediction.

## 📈 Project Statistics

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 84.45% |
| **Training Samples** | 111,778 |
| **Data Sources** | 4 (Gaming, Mobile, Twitter, Reviews) |
| **Languages** | 2 (English, Arabic) |
| **Response Time** | <1 second |
| **API Endpoints** | 6 |

---

## 🛠️ Technology Stack

### Backend
- **Framework**: Django 4.2.7
- **Language**: Python 3.9+
- **Database**: SQLite (dev) / PostgreSQL (prod-ready)

### Machine Learning
- **Algorithm**: Random Forest Classifier (100 estimators)
- **Vectorization**: TF-IDF (5,000 features, n-grams 1-2)
- **Libraries**: scikit-learn, pandas, NumPy
- **Training Data**: 111K+ labeled samples

### Frontend
- **Framework**: Bootstrap 5
- **Charts**: Chart.js
- **Icons**: Font Awesome 6
- **Animations**: CSS3 + JavaScript

### Translation
- **Engine**: Google Translate API (googletrans)
- **Auto-Detection**: Language identification
- **Supported**: Arabic ↔ English

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip (Python package manager)
Virtual environment (recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run migrations**
```bash
python manage.py migrate
```

5. **Start the server**
```bash
python manage.py runserver
```

6. **Access the application**
```
http://127.0.0.1:8000/
```

---

## 📁 Project Structure

```
sentiment_project/
├── manage.py
├── requirements.txt
├── README.md
├── ml_models/
│   ├── sentiment_model.pkl      # Trained Random Forest model
│   └── vectorizer.pkl            # TF-IDF vectorizer
├── sentiment_project/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── sentiment_app/
│   ├── models.py                 # Database models
│   ├── views.py                  # Application logic
│   ├── urls.py                   # URL routing
│   ├── forms.py                  # Form definitions
│   ├── admin.py                  # Admin interface
│   └── templates/
│       ├── base.html
│       ├── home.html
│       ├── result.html
│       ├── history.html
│       ├── analytics.html
│       └── landing.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
└── data/
    └── processed_data.csv        # Training dataset
```

---

## 🎯 API Documentation

### Base URL
```
http://your-domain.com/api/
```

### Endpoints

#### 1. Predict Sentiment (Legacy)
```http
POST /api/predict/
Content-Type: application/json

{
  "text": "This product is amazing!"
}
```

**Response:**
```json
{
  "success": true,
  "text": "This product is amazing!",
  "language": "english",
  "sentiment": "positive",
  "confidence": 89.2,
  "probabilities": {
    "positive": 0.892,
    "negative": 0.054,
    "neutral": 0.054
  }
}
```

#### 2. Predict Sentiment (DRF v2 JSON)
```http
POST /api/v2/predict/
Content-Type: application/json

{
  "text": "This product is amazing!"
}
```

#### 3. Arabic Text Example
```http
POST /api/predict/
Content-Type: application/json

{
  "text": "المنتج رائع جداً!"
}
```

**Response:**
```json
{
  "success": true,
  "text": "المنتج رائع جداً!",
  "language": "arabic",
  "was_translated": true,
  "translated_text": "The product is very wonderful!",
  "sentiment": "positive",
  "confidence": 91.5,
  "probabilities": {
    "positive": 0.915,
    "negative": 0.043,
    "neutral": 0.042
  }
}
```

---

## 📊 Model Performance

### Training Metrics
```
Algorithm: Random Forest Classifier
Training Samples: 89,267 (80%)
Test Samples: 22,317 (20%)
Features: 5,000 (TF-IDF)
```

### Classification Report
```
              precision    recall  f1-score   support

    negative       0.89      0.78      0.83      5823
     neutral       0.83      0.87      0.85      8851
    positive       0.83      0.86      0.84      7643

    accuracy                           0.84     22317
   macro avg       0.85      0.84      0.84     22317
weighted avg       0.85      0.84      0.84     22317
```

### Model Comparison
| Algorithm | Accuracy |
|-----------|----------|
| **Random Forest** | **84.45%** ✓ |
| Logistic Regression | 69.80% |
| Linear SVC | 69.58% |
| Naive Bayes | 64.27% |

---

## 🎨 Features Showcase

### 1. Multi-Language Analysis
```python
# English
Input: "This is fantastic!"
Output: Positive (87% confidence)

# Arabic
Input: "هذا رائع جداً!"
Translation: "This is very wonderful!"
Output: Positive (91% confidence)
```

### 2. Real-Time Analytics
- Pie chart showing sentiment distribution
- Line chart showing 7-day trends
- Key insights and recommendations
- Exportable data

### 3. History Tracking
- Filterable by sentiment (All/Positive/Negative/Neutral)
- Search functionality
- Expandable text preview
- Confidence visualization

---

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///db.sqlite3
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Production Settings
```python
# settings.py
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'sentiment_db',
        'USER': 'dbuser',
        'PASSWORD': 'dbpassword',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

---

## 📚 Data Sources

The model was trained on diverse datasets:
1. **Gaming Comments** (23,189 samples) - Reddit gaming discussions
2. **Mobile Reviews** (50,000 samples) - Product reviews
3. **Twitter Data** (74,681 samples) - Social media posts
4. **Cleaned Comments** (21,821 samples) - Preprocessed text

**Total**: 111,778 labeled samples after preprocessing

---

## 🧪 Testing

### Run Tests
```bash
python manage.py test
```

### Test Coverage
```bash
coverage run --source='.' manage.py test
coverage report
```

### Manual Testing
```python
python manage.py shell

>>> from sentiment_app.models import SentimentPrediction
>>> predictions = SentimentPrediction.objects.all()
>>> print(f"Total: {predictions.count()}")
```

---

## 🚢 Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "sentiment_project.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### Heroku
```bash
heroku create your-app-name
git push heroku main
heroku run python manage.py migrate
```

### AWS/Azure
- Use Elastic Beanstalk or App Service
- Configure environment variables
- Set up PostgreSQL database
- Enable static files serving

---

## 📈 Future Enhancements

- [ ] Deep Learning models (LSTM, BERT)
- [ ] More languages support (French, Spanish, German)
- [ ] Emotion detection (Joy, Anger, Fear, etc.)
- [ ] Batch processing for CSV uploads
- [ ] WebSocket for real-time analysis
- [ ] Mobile app (React Native)
- [ ] Chrome extension
- [ ] Slack/Discord bot integration

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-linkedin](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset sources: Reddit, Twitter, Mobile Reviews
- scikit-learn community for ML algorithms
- Django community for the web framework
- Bootstrap team for the UI framework
- Chart.js for data visualization

---

## 📞 Support

If you have any questions or need help:
- 📧 Email: your.email@example.com
- 💬 Open an issue on GitHub
- 🌐 Visit our website

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ by [Your Name]**

*Powered by Django + scikit-learn + Bootstrap*
