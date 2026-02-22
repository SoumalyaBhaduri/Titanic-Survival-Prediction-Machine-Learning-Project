import joblib
import numpy as np
from app.config import settings
from app.utils import setup_logger

logger = setup_logger()


class ModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load(settings.MODEL_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, data):
        try:
            input_array = np.array(
                [[data.Pclass, data.Sex, data.Age, data.SibSp, data.Parch, data.Fare]]
            )

            prediction = self.model.predict(input_array)[0]
            probability = self.model.predict_proba(input_array)[0][1]

            return int(prediction), float(probability)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


model_service = ModelService()
