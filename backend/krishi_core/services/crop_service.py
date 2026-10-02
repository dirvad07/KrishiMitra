import os
import logging
import joblib

logger = logging.getLogger("core.crop_service")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml_models",
    "artifacts",
    "crop_recommendation_model.pkl",
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "ml_models",
    "artifacts",
    "crop_label_encoder.pkl",
)


class CropRecommendationService:

    def __init__(self):
        self.model = None
        self.encoder = None
        self._loaded = False

        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Crop recommendation model not found: {MODEL_PATH}")
                return

            if not os.path.exists(ENCODER_PATH):
                logger.warning(f"Crop label encoder not found: {ENCODER_PATH}")
                return

            self.model = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            self._loaded = True

            logger.info("Crop recommendation model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load crop recommendation model: {e}")

    def predict(self, data):

        if not self._loaded:
            return "N/A (Model unavailable)"

        try:
            features = [[
                data["N"],
                data["P"],
                data["K"],
                data["temperature"],
                data["humidity"],
                data["ph"],
                data["rainfall"],
            ]]

            prediction = self.model.predict(features)[0]

            crop = self.encoder.inverse_transform([prediction])[0]

            return crop

        except Exception as e:
            logger.error(f"Crop prediction error: {e}")
            return "Error (Prediction failed)"


crop_service = CropRecommendationService()