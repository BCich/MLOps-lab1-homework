"""Unit tests for the sentiment analysis predict endpoint."""

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    """Create a test client with loaded ML models."""
    with TestClient(app) as client:
        yield client


class TestInputValidation:
    """Tests for request input validation."""

    def test_valid_input(self, client):
        """Valid non-empty text should return 200."""
        response = client.post("/predict", json={"text": "I love this"})
        assert response.status_code == 200

    def test_empty_string(self, client):
        """Empty string should be rejected with 422 and error details."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_missing_text_field(self, client):
        """Missing 'text' field should be rejected with 422 and error details."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_wrong_type(self, client):
        """Non-string 'text' value should be rejected with 422."""
        response = client.post("/predict", json={"text": 123})
        assert response.status_code == 422


class TestModelInference:
    """Tests for model inference on sample texts."""

    def test_positive_sentiment(self, client):
        """Positive text should return a valid sentiment label."""
        response = client.post("/predict", json={"text": "This is amazing, I love it!"})
        assert response.status_code == 200
        assert response.json()["prediction"] in ["positive", "negative", "neutral"]

    def test_negative_sentiment(self, client):
        """Negative text should return a valid sentiment label."""
        response = client.post("/predict", json={"text": "This is terrible, I hate it"})
        assert response.status_code == 200
        assert response.json()["prediction"] in ["positive", "negative", "neutral"]

    def test_neutral_sentiment(self, client):
        """Neutral text should return a valid sentiment label."""
        response = client.post("/predict", json={"text": "The meeting is at 3pm"})
        assert response.status_code == 200
        assert response.json()["prediction"] in ["positive", "negative", "neutral"]


class TestResponseValidation:
    """Tests for response format validation."""

    def test_response_has_prediction_key(self, client):
        """Response JSON should contain 'prediction' key with string value."""
        response = client.post("/predict", json={"text": "Hello world"})
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], str)

    def test_response_is_valid_json(self, client):
        """Response should have JSON content type and be parseable."""
        response = client.post("/predict", json={"text": "Test"})
        assert response.headers["content-type"] == "application/json"
        response.json()

    def test_invalid_input_returns_json_error(self, client):
        """Invalid input should return JSON error response with details."""
        response = client.post("/predict", json={"text": ""})
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert "detail" in data
