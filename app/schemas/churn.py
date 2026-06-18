from enum import Enum
from pydantic import BaseModel, Field

class ContractEnum(str, Enum):
    """Contract terms for customer."""
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class InternetServiceEnum(str, Enum):
    """Internet service providers."""
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class PaymentMethodEnum(str, Enum):
    """Payment methods used by the customer."""
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"


class ChurnInput(BaseModel):
    """
    Pydantic Schema for Telco Customer Churn Prediction Request.
    Provides strict type validation and structural compliance.
    """
    # Strict validation of prompt-specified key features
    tenure: int = Field(
        ..., 
        ge=0, 
        description="Number of months the customer has stayed with the company."
    )
    MonthlyCharges: float = Field(
        ..., 
        gt=0.0, 
        description="The amount charged to the customer monthly."
    )
    TotalCharges: float = Field(
        ..., 
        ge=0.0, 
        description="The total amount charged to the customer."
    )
    Contract: ContractEnum = Field(
        ..., 
        description="The contract term of the customer (Month-to-month, One year, Two year)."
    )
    total_services: int = Field(
        ..., 
        ge=0, 
        description="Total number of services signed up by the customer."
    )

    # Optional / Defaulted features to feed the XGBoost 24-feature space
    SeniorCitizen: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer is a senior citizen (1) or not (0)."
    )
    Partner: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has a partner (1) or not (0)."
    )
    Dependents: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has dependents (1) or not (0)."
    )
    OnlineSecurity: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has Online Security (1) or not (0)."
    )
    OnlineBackup: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has Online Backup (1) or not (0)."
    )
    DeviceProtection: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has Device Protection (1) or not (0)."
    )
    TechSupport: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has Tech Support (1) or not (0)."
    )
    StreamingTV: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has Streaming TV (1) or not (0)."
    )
    StreamingMovies: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has Streaming Movies (1) or not (0)."
    )
    PaperlessBilling: int = Field(
        0, 
        ge=0, 
        le=1, 
        description="Whether the customer has Paperless Billing (1) or not (0)."
    )
    InternetService: InternetServiceEnum = Field(
        InternetServiceEnum.NO, 
        description="Type of Internet Service."
    )
    PaymentMethod: PaymentMethodEnum = Field(
        PaymentMethodEnum.MAILED_CHECK, 
        description="Payment Method used by the customer."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.05,
                "TotalCharges": 840.60,
                "Contract": "One year",
                "total_services": 3,
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "OnlineSecurity": 1,
                "OnlineBackup": 0,
                "DeviceProtection": 1,
                "TechSupport": 1,
                "StreamingTV": 0,
                "StreamingMovies": 0,
                "PaperlessBilling": 1,
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check"
            }
        }
    }


class ChurnOutput(BaseModel):
    """
    Pydantic Schema for Telco Customer Churn Prediction Response.
    """
    success: bool = Field(True, description="Indicates if prediction was processed successfully.")
    churn_prediction: int = Field(..., description="Predicted Churn class (1 = Churn, 0 = Active Customer).")
    churn_probability: float = Field(..., description="Probability score of the customer churning (range 0.0 - 1.0).")
    model_version: str = Field("XGBoost v1.0", description="The registered model version running this inference.")
