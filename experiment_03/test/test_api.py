from fastapi.testclient import TestClient
from experiment_03.src.app_fast import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200