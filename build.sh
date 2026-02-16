#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Retrain model if it doesn't exist
MODEL_PATH="ml_models/sentiment_model.pkl"
VECTORIZER_PATH="ml_models/vectorizer.pkl"

# Optional retraining: disabled by default on deploy to avoid long/fragile builds.
# Enable explicitly with RETRAIN_ON_BUILD=1 when you really want to retrain in CI.
if [ ! -f "$MODEL_PATH" ] || [ ! -f "$VECTORIZER_PATH" ]; then
    echo "⚠️ Pretrained model artifacts were not found."
    if [ "${RETRAIN_ON_BUILD:-0}" = "1" ]; then
        echo "RETRAIN_ON_BUILD=1 -> retraining model now..."
        python retrain_model.py
    else
        echo "Skipping retraining during build (RETRAIN_ON_BUILD!=1)."
        echo "The app will still boot, but prediction endpoints will return model-not-loaded errors until model files exist."
    fi
fi

# Collect static files
python manage.py collectstatic --no-input

# Run migrations
echo "Running migrations..."
python manage.py migrate

echo "Build process completed successfully!"
