import os
import logging
import joblib
import pandas as pd
from django.conf import settings

logger = logging.getLogger("core.yield_service")

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "krishi_core",
    "ml_models",
    "artifacts",
    "crop_yield_model.pkl",
)


class CropYieldPredictionService:

    def __init__(self):
        self.model = None
        self._loaded = False

        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(
                    f"Crop yield model not found: {MODEL_PATH}"
                )
                return

            self.model = joblib.load(MODEL_PATH)
            self._loaded = True

            logger.info(
                "Crop yield prediction model loaded successfully."
            )

        except Exception as e:
            logger.error(
                f"Failed to load crop yield model: {e}"
            )

    def predict(self, data):

        if not self._loaded:
            return "N/A (Model unavailable)"

        try:
            features = pd.DataFrame([{
                "Crop": data["Crop"],
                "Crop_Year": data["Crop_Year"],
                "Season": data["Season"],
                "State": data["State"],
                "Area": data["Area"],
                "Annual_Rainfall": data["Annual_Rainfall"],
                "Fertilizer": data["Fertilizer"],
                "Pesticide": data["Pesticide"],
            }])

            prediction = self.model.predict(features)[0]

            return round(float(prediction), 2)

        except Exception as e:
            logger.error(
                f"Crop yield prediction error: {e}"
            )
            return "Error (Prediction failed)"


yield_service = CropYieldPredictionService()