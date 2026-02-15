"""
Retrain Sentiment Analysis Model
This will retrain the model using your current scikit-learn version
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING SENTIMENT MODEL FOR YOUR ENVIRONMENT")
print("="*80)

# Check scikit-learn version
import sklearn
print(f"\nYour scikit-learn version: {sklearn.__version__}")

# Load processed data
print("\n[1] Loading processed data...")
try:
    # Try multiple paths for data
    data_paths = ['processed_data.csv', 'data/processed_data.csv', '../data/processed_data.csv']
    df = None
    for path in data_paths:
        if os.path.exists(path):
            print(f"Found data at: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("processed_data.csv not found in any expected location")
    print(f"✅ Data loaded: {df.shape}")
except FileNotFoundError:
    print("❌ processed_data.csv not found!")
    print("Please make sure processed_data.csv is in the current directory")
    exit(1)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

print("\n[2] Cleaning text data...")
df['cleaned_text'] = df['text'].apply(clean_text)
df = df[df['cleaned_text'].str.len() > 0]
print(f"✅ Data cleaned: {df.shape}")

# Split data
print("\n[3] Splitting data...")
X = df['cleaned_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training set: {X_train.shape}")
print(f"✅ Test set: {X_test.shape}")

# Vectorize
print("\n[4] Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"✅ Feature matrix shape: {X_train_vec.shape}")

# Train model
print("\n[5] Training Random Forest model...")
print("This may take 2-3 minutes...")

model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_vec, y_train)
print("✅ Model trained!")

# Evaluate
print("\n[6] Evaluating model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*80}")
print(f"MODEL ACCURACY: {accuracy:.4f}")
print(f"{'='*80}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save models
print("\n[7] Saving models...")
os.makedirs('ml_models', exist_ok=True)
joblib.dump(model, 'ml_models/sentiment_model.pkl')
joblib.dump(vectorizer, 'ml_models/vectorizer.pkl')
print("✅ Models saved to ml_models/")

# Test predictions
print("\n[8] Testing predictions...")
test_texts = [
    "This game is absolutely amazing! I love it!",
    "Worst experience ever. Completely disappointed.",
    "It's okay, nothing special."
]

for text in test_texts:
    cleaned = clean_text(text)
    text_vec = vectorizer.transform([cleaned])
    prediction = model.predict(text_vec)[0]
    confidence = model.predict_proba(text_vec)[0].max() * 100
    print(f"\nText: {text}")
    print(f"Prediction: {prediction} ({confidence:.1f}%)")

print("\n" + "="*80)
print("RETRAINING COMPLETE!")
print("="*80)
print("\n✅ New models saved and ready to use!")
print("✅ Now restart your Django server: python manage.py runserver")
