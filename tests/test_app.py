from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
def test_health():
    r=client.get("/health")
    assert r.status_code==200 and "model_available" in r.json()
def test_prediction_requires_model_or_returns_score():
    r=client.post("/predict",json={"temperature_c":70,"vibration_mm_s":2,"pressure_bar":5.5,"operating_hours":1000})
    assert r.status_code in (200,503)
