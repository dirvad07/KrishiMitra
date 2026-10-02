import os
import logging
import joblib
import pandas as pd
from django.conf import settings

logger = logging.getLogger("core.fertilizer_service")

MODEL_DIR = os.path.join(
    settings.BASE_DIR,
    "krishi_core",
    "ml_models",
    "artifacts"
)

MODEL_PATH = os.path.join(MODEL_DIR, "fertilizer_model.pkl")
PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "fertilizer_feature_preprocessor.pkl"
)
LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "fertilizer_label_encoder.pkl"
)


class FertilizerRecommendationService:

    def __init__(self):
        self.model = None
        self.encoder = None
        self._loaded = False

        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(
                    f"Fertilizer model not found: {MODEL_PATH}"
                )
                return

            if not os.path.exists(LABEL_ENCODER_PATH):
                logger.warning(
                    f"Fertilizer label encoder not found: "
                    f"{LABEL_ENCODER_PATH}"
                )
                return

            self.model = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(LABEL_ENCODER_PATH)

            self._loaded = True

            logger.info(
                "Fertilizer recommendation model loaded successfully."
            )

        except Exception as e:
            logger.error(
                f"Failed to load fertilizer model: {e}"
            )

    def predict(self, data):

        if not self._loaded:
            return "N/A (Model unavailable)"

        try:
            features = pd.DataFrame([{
                "Soil_Type": data.get("Soil_Type", "Black"),
                "Crop_Type": data.get("Crop_Type", "Wheat"),
                "Crop_Growth_Stage": data.get(
                    "Crop_Growth_Stage",
                    "Pre-emergence"
                ),
                "Season": data.get("Season", "Kharif"),
                "Irrigation_Type": data.get("Irrigation_Type", "Drip"),
                "Previous_Crop": data.get("Previous_Crop", "Unknown"),
                "Region": data.get("Region", "Maharashtra"),
                "Soil_pH": float(data.get("Soil_pH", 6.5)),
                "Soil_Moisture": float(data.get("Soil_Moisture", 40.0)),
                "Organic_Carbon": float(data.get("Organic_Carbon", 0.5)),
                "Electrical_Conductivity": float(
                    data.get("Electrical_Conductivity", 0.4)
                ),
                "Nitrogen_Level": float(
                    data.get("Nitrogen_Level", 100)
                ),
                "Phosphorus_Level": float(
                    data.get("Phosphorus_Level", 30)
                ),
                "Potassium_Level": float(
                    data.get("Potassium_Level", 200)
                ),
                "Temperature": float(
                    data.get("Temperature", 25.0)
                ),
                "Humidity": float(
                    data.get("Humidity", 60.0)
                ),
                "Rainfall": float(
                    data.get("Rainfall", 100.0)
                ),
                "Fertilizer_Used_Last_Season": data.get(
                    "Fertilizer_Used_Last_Season",
                    "None"
                ),
                "Yield_Last_Season": float(
                    data.get("Yield_Last_Season", 0.0)
                ),
            }])

            prediction = self.model.predict(features)[0]

            if self.encoder and hasattr(
                self.encoder,
                "inverse_transform"
            ):
                try:
                    return self.encoder.inverse_transform(
                        [prediction]
                    )[0]
                except Exception:
                    return str(prediction)

            return str(prediction)

        except Exception as e:
            logger.error(
                f"Fertilizer prediction error: {e}"
            )
            return "Error (Prediction failed)"


fertilizer_service = FertilizerRecommendationService()