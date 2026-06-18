import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    PROJECT_NAME: str = "Telco Customer Churn Prediction API"
    API_V1_STR: str = "/api/v1"
    
    # Model configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/models/xgb_churn_model.pkl")
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
