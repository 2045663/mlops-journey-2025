from fastapi.testclient import TestClient
from ..src.app_fast import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # 验证失败是预期行为