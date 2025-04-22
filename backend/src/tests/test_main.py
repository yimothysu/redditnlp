import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock environment variables
with patch.dict(os.environ, {
    "MONGODB_CONNECTION_STRING": "mongodb://testuser:testpass@localhost:27017/testdb",
    "REDDIT_CLIENT_ID": "test_client_id", 
    "REDDIT_CLIENT_SECRET": "test_client_secret"
}):
    sys.modules['src.database.db'] = MagicMock()
    sys.modules['src.data_fetchers.toxicity_fetcher'] = MagicMock()
    sys.modules['src.data_fetchers.positive_content_fetcher'] = MagicMock()

from main import app

client = TestClient(app)

def test_analysis_invalid_time_filter():
    """
    Tests that the API rejects requests with invalid time filters.
    """
    response = client.post(
        "/analysis/",
        json={
            "name": "AskReddit",
            "sort_by": "top",
            "time_filter": "decade"
        }
    )
    assert response.status_code == 400
    assert "Invalid time filter" in response.json()["detail"]


def test_analysis_invalid_sort_method():
    """
    Tests that the API rejects requests with invalid sort methods.
    """
    response = client.post(
        "/analysis/",
        json={
            "name": "AskReddit",
            "sort_by": "trending",
            "time_filter": "week"
        }
    )
    assert response.status_code == 400
    assert "Invalid sort method" in response.json()["detail"]
