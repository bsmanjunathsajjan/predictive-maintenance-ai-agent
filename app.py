"""Predictive maintenance risk API with human-approved recommendations."""
import os, joblib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
app=FastAPI(title="Predictive Maintenance AI Agent",version="1.0.0")
MODEL_PATH=os.getenv("MODEL_PATH","maintenance_model.joblib")
class SensorReading(BaseModel):
    temperature_c: float=Field(ge=-20,le=250)
    vibration_mm_s: float=Field(ge=0,le=100)
    pressure_bar: float=Field(ge=0,le=100)
    operating_hours: int=Field(ge=0)
@app.get("/health")
def health(): return {"status":"ok","model_available":os.path.exists(MODEL_PATH)}
@app.post("/predict")
def predict(r:SensorReading):
    if not os.path.exists(MODEL_PATH): raise HTTPException(503,"Model missing. Run python train.py first.")
    model=joblib.load(MODEL_PATH)
    x=np.array([[r.temperature_c,r.vibration_mm_s,r.pressure_bar,r.operating_hours]])
    p=float(model.predict_proba(x)[0,1])
    risk="high" if p>=.7 else "medium" if p>=.35 else "low"
    advice={"high":"Arrange prompt inspection by a qualified operator.","medium":"Increase monitoring and plan preventive inspection.","low":"Continue routine monitoring and scheduled maintenance."}[risk]
    return {"failure_probability":round(p,4),"risk_level":risk,"recommendation":advice,"human_approval_required":True}
