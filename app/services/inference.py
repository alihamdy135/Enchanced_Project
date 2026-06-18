import os
import pickle
import pandas as pd
import numpy as np
from typing import Tuple
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ModelLoadError, ModelInferenceError
from app.schemas.churn import ChurnInput

class ChurnInferenceService:
    """
    ML Service coordinating pre-trained model loading, feature mapping,
    and running real-time churn predictions.
    """
    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = model_path
        self._model = None
        
        # Exact features order required by the pre-trained XGBoost Classifier
        self.expected_features = [
            'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
            'total_services', 'InternetService_DSL', 'InternetService_Fiber optic',
            'InternetService_No', 'Contract_Month-to-month', 'Contract_One year',
            'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
        ]

    def _load_model(self) -> None:
        """
        Lazily loads the pickle model binary into memory.
        """
        if self._model is not None:
            return
        
        if not os.path.exists(self.model_path):
            logger.critical(f"Model file not found at path: {self.model_path}")
            raise ModelLoadError(f"Model binary not found at: {self.model_path}")

        try:
            logger.info(f"Loading XGBoost Churn Model from: {self.model_path}")
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("XGBoost Churn Model successfully loaded into memory.")
        except Exception as e:
            logger.exception(f"Error occurred while loading the pickle model: {str(e)}")
            raise ModelLoadError(f"Failed to load XGBoost Churn Model binary: {str(e)}")

    def preprocess_features(self, data: ChurnInput) -> pd.DataFrame:
        """
        Transforms the high-level ChurnInput schema into the structured,
        one-hot encoded DataFrame expected by the model.
        """
        try:
            # Initialize feature dict with zeros
            row = {col: 0.0 for col in self.expected_features}
            
            # Map prompt-specified features
            row['tenure'] = float(data.tenure)
            row['MonthlyCharges'] = float(data.MonthlyCharges)
            row['TotalCharges'] = float(data.TotalCharges)
            row['total_services'] = float(data.total_services)

            # Map other features directly
            row['SeniorCitizen'] = float(data.SeniorCitizen)
            row['Partner'] = float(data.Partner)
            row['Dependents'] = float(data.Dependents)
            row['OnlineSecurity'] = float(data.OnlineSecurity)
            row['OnlineBackup'] = float(data.OnlineBackup)
            row['DeviceProtection'] = float(data.DeviceProtection)
            row['TechSupport'] = float(data.TechSupport)
            row['StreamingTV'] = float(data.StreamingTV)
            row['StreamingMovies'] = float(data.StreamingMovies)
            row['PaperlessBilling'] = float(data.PaperlessBilling)

            # Map 'Contract' categorical value
            contract_key = f"Contract_{data.Contract.value}"
            if contract_key in row:
                row[contract_key] = 1.0

            # Map 'InternetService' categorical value
            internet_key = f"InternetService_{data.InternetService.value}"
            if internet_key in row:
                row[internet_key] = 1.0

            # Map 'PaymentMethod' categorical value
            payment_key = f"PaymentMethod_{data.PaymentMethod.value}"
            if payment_key in row:
                row[payment_key] = 1.0

            # Construct DataFrame and enforce column order
            df = pd.DataFrame([row])
            df = df[self.expected_features]
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering/preprocessing failed: {str(e)}")
            raise ModelInferenceError(f"Failed to preprocess features: {str(e)}")

    def predict(self, data: ChurnInput) -> Tuple[int, float]:
        """
        Runs model preprocessing and makes a prediction.
        Returns:
            churn_prediction (int): 0 or 1.
            churn_probability (float): float between 0 and 1.
        """
        # Ensure model is loaded
        self._load_model()
        
        # Preprocess input data
        features_df = self.preprocess_features(data)
        
        try:
            logger.info("Executing XGBoost model predict_proba...")
            
            # Predict probability of churn (class 1)
            probabilities = self._model.predict_proba(features_df)
            churn_prob = float(probabilities[0][1])
            
            # Predict binary class
            predictions = self._model.predict(features_df)
            churn_pred = int(predictions[0])
            
            logger.info(f"Prediction success: Churn={churn_pred}, Probability={churn_prob:.4f}")
            return churn_pred, churn_prob
            
        except Exception as e:
            logger.exception(f"Inference execution failed: {str(e)}")
            raise ModelInferenceError(f"Failed to execute model prediction: {str(e)}")

# Singleton instance of the inference service
inference_service = ChurnInferenceService()
