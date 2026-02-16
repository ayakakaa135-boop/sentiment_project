from unittest.mock import patch

from django.test import TestCase

from .views import clean_text, normalize_sentiment_label


class TextPreprocessingTests(TestCase):
    def test_clean_text_removes_links_and_symbols(self):
        text = 'Amazing!!! Visit https://example.com now'
        cleaned = clean_text(text)
        self.assertNotIn('http', cleaned)
        self.assertIn('amaz', cleaned)

    def test_normalize_sentiment_label_handles_transformers_labels(self):
        self.assertEqual(normalize_sentiment_label('LABEL_2'), 'positive')
        self.assertEqual(normalize_sentiment_label('label_0'), 'negative')


class APIV2Tests(TestCase):
    @patch('sentiment_app.views.get_prediction_payload')
    @patch('sentiment_app.views.load_models')
    def test_api_v2_predict_success(self, mock_load_models, mock_get_payload):
        mock_load_models.return_value = True
        mock_get_payload.return_value = {
            'text': 'Great app',
            'language': 'english',
            'was_translated': False,
            'translated_text': None,
            'cleaned_text': 'great app',
            'sentiment': 'positive',
            'confidence': 93.4,
            'probabilities': {'positive': 0.934},
        }

        response = self.client.post(
            '/en/api/v2/predict/',
            data={'text': 'Great app'},
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        self.assertEqual(response.json()['sentiment'], 'positive')

    def test_api_v2_requires_text(self):
        response = self.client.post(
            '/en/api/v2/predict/',
            data={},
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
