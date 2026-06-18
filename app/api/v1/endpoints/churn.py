from fastapi import APIRouter, status
from app.schemas.churn import ChurnInput, ChurnOutput
from app.services.inference import inference_service
from app.core.logging import logger

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict:
    """
    GET Health Check Endpoint.
    Verifies FastAPI server status and model loading integrity.
    """
    try:
        # Preemptively check if the model loads successfully
        inference_service._load_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "message": "FastAPI service is running and XGBoost model is loaded successfully."
        }
    except Exception as e:
        logger.exception(f"Health check found system in DEGRADED status: {str(e)}")
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e),
            "detail": "Failed to load pre-trained XGBoost model binary."
        }


@router.post("/predict", response_model=ChurnOutput, status_code=status.HTTP_200_OK)
def predict_churn(payload: ChurnInput) -> ChurnOutput:
    """
    POST Prediction Endpoint.
    Receives validated customer features, executes XGBoost model inference,
    and returns a structured churn probability and classification.
    """
    logger.info("Received request on churn prediction API endpoint.")
    
    # Run prediction (exceptions bubble up and are caught by FastAPI custom middleware)
    churn_pred, churn_prob = inference_service.predict(payload)
    
    return ChurnOutput(
        success=True,
        churn_prediction=churn_pred,
        churn_probability=churn_prob,
        model_version="XGBoost Churn Classifier v1.0"
    )
