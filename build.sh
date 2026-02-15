#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Retrain model if it doesn't exist
if [ ! -f "ml_models/sentiment_model.pkl" ]; then
    echo "Model not found, retraining..."
    python retrain_model.py
fi

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py migrate
