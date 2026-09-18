from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    transaction = {
        "Time": 406,
        "Amount": 123.50
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "fraud_probability" in data
    assert "decision_threshold" in data