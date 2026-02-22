from fastapi import FastAPI, HTTPException
from app.schema import Passenger, PredictionResponse
from app.model import model_service
from app.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: Passenger):
    try:
        prediction, probability = model_service.predict(data)
        return PredictionResponse(
            survived=prediction, probability=round(probability, 4)
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
