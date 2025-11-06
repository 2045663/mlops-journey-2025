from fastapi.testclient import TestClient
from ..src.app_fast import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def get_token():
    response = client.post("/token")
    return response.json()["access_token"]

def test_predict_missing_field():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/predict", json={}, headers=headers)
    assert response.status_code == 422  # 验证失败是预期行为