import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
import csv
import os
from datetime import datetime

# Initialize FastAPI App
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Production MLOps pipeline for Customer Churn classification using XGBoost.",
    version="1.0.0",
    docs_url=None  # Disable default OpenAPI Swagger docs
)


# CORS Policy configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared directories & paths setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "xgb_churn_model.pkl")

# Lazy load model container
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model file not found at: {MODEL_PATH}")
    
    try:
        _model = joblib.load(MODEL_PATH)
        # Apply version-compatibility patch for model parameters
        import inspect
        params = set()
        for cls in inspect.getmro(_model.__class__):
            if hasattr(cls, '_get_param_names'):
                params.update(cls._get_param_names())
        for p in params:
            if not hasattr(_model, p):
                setattr(_model, p, None)
        return _model
    except Exception as e:
        raise RuntimeError(f"Error loading model binary: {str(e)}")


# 1. Pydantic Validation Schema (Exactly 5 required features)
class ChurnInput(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months customer has stayed with company")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges billed to customer")
    TotalCharges: float = Field(..., ge=0, description="Total charges accumulated by customer")
    Contract: int = Field(..., ge=0, le=2, description="0: Month-to-month, 1: One year, 2: Two year")
    total_services: int = Field(..., ge=0, description="Number of subscribed auxiliary services")


# 2. API Endpoints

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Serves custom themed Swagger UI docs."""
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css"
    )
    # Inject the theme CSS right before the closing head tag
    html_content = response.body.decode("utf-8")
    themed_html = html_content.replace(
        "</head>",
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-themes@3.0.1/themes/3.x/theme-material.css"></head>'
    )
    return HTMLResponse(content=themed_html, status_code=response.status_code)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_gui():
    """Serves the premium custom black-themed single page dashboard & predictor GUI."""
    gui_path = os.path.join(CURRENT_DIR, "core", "gui.html")
    if not os.path.exists(gui_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GUI HTML asset not found locally."
        )
    with open(gui_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/health")
async def check_health():
    """Health check endpoint validating API operational status and XGBoost model loading."""
    try:
        load_model()
        return {
            "status": "healthy",
            "message": "XGBoost customer churn model is fully loaded and operational.",
            "model_path": MODEL_PATH
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": f"Model load failure: {str(e)}"
        }


@app.post("/predict")
async def predict_churn(data: ChurnInput):
    """
    Executes real-time MLOps prediction.
    Accepts 5 core customer features and dynamically maps them to the 24 one-hot encoded features
    required by the pre-trained XGBoost Classifier.
    """
    try:
        # Enforce XGBoost model load
        model = load_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Predictive engine is offline: {str(e)}"
        )

    # All 24 features expected by our pre-trained model in exact order
    expected_features = [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'total_services', 'InternetService_DSL', 'InternetService_Fiber optic',
        'InternetService_No', 'Contract_Month-to-month', 'Contract_One year',
        'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]

    try:
        # Transform Pydantic inputs and default other engineered features
        row = {col: 0.0 for col in expected_features}
        
        # Populate 5 core parameters
        row['tenure'] = float(data.tenure)
        row['MonthlyCharges'] = float(data.MonthlyCharges)
        row['TotalCharges'] = float(data.TotalCharges)
        row['total_services'] = float(data.total_services)

        # Map Contract int representation to one-hot encoding columns
        # Contract values: 0 = Month-to-month, 1 = One year, 2 = Two year
        if data.Contract == 0:
            row['Contract_Month-to-month'] = 1.0
        elif data.Contract == 1:
            row['Contract_One year'] = 1.0
        elif data.Contract == 2:
            row['Contract_Two year'] = 1.0

        # Set sensible standard baseline default parameters for auxiliary columns
        # Since these aren't validated by the minimal schema, we default them
        row['SeniorCitizen'] = 0.0
        row['Partner'] = 0.0
        row['Dependents'] = 0.0
        row['OnlineSecurity'] = 0.0
        row['OnlineBackup'] = 0.0
        row['DeviceProtection'] = 0.0
        row['TechSupport'] = 0.0
        row['StreamingTV'] = 0.0
        row['StreamingMovies'] = 0.0
        row['PaperlessBilling'] = 1.0  # Common default

        # Dynamic Default mapping for InternetService (DSL, Fiber optic, No)
        row['InternetService_DSL'] = 0.0
        row['InternetService_Fiber optic'] = 1.0  # Common baseline
        row['InternetService_No'] = 0.0

        # Dynamic Default mapping for PaymentMethod (Electronic check, Mailed check, Credit card, Bank transfer)
        row['PaymentMethod_Bank transfer (automatic)'] = 0.0
        row['PaymentMethod_Credit card (automatic)'] = 0.0
        row['PaymentMethod_Electronic check'] = 1.0  # Common baseline
        row['PaymentMethod_Mailed check'] = 0.0

        # Build Pandas input matrix matching features schema
        df = pd.DataFrame([row])
        df = df[expected_features]

        # Convert to float explicit to avoid type errors
        for col in expected_features:
            df[col] = df[col].astype(float)

        # Execute predictions
        churn_prob = float(model.predict_proba(df)[0][1])
        churn_pred = int(model.predict(df)[0])
        # ==============================
        # Prediction Logging to CSV
        # ==============================

        log_dir = os.path.join(CURRENT_DIR, "data", "predictions")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "prediction_logs.csv")

        file_exists = os.path.isfile(log_file)

        with open(log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # Create headers if file doesn't exist
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "Contract",
                    "total_services",
                    "churn_prediction",
                    "churn_probability"
                ])

            # Write prediction row
            writer.writerow([
                datetime.now(),
                data.tenure,
                data.MonthlyCharges,
                data.TotalCharges,
                data.Contract,
                data.total_services,
                churn_pred,
                churn_prob
            ])

        return {
            "churn_prediction": churn_pred,
            "churn_probability": churn_prob,
            "model_version": "XGBoost Churn v1.0",
            "features_mapped": 24
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )