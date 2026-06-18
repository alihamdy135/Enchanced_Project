import os
import joblib
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Try importing Airflow components, fallback to Mock classes if run in non-Airflow context
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    class DAG:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    class MockOperator:
        def __init__(self, *args, **kwargs): pass
        def __rshift__(self, other): return other
    def PythonOperator(*args, **kwargs): return MockOperator()

# Configure logging for the pipeline execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airflow.dag.churn_inference")

# Default arguments for Airflow Scheduler
default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# Container local directories (mounted to host data folder)
RAW_DATA_PATH = "/app/data/inputs/raw_daily_data.csv"
ENGINEERED_DATA_PATH = "/app/data/inputs/engineered_daily_data.csv"
RESULTS_PATH = "/app/data/predictions/daily_results.csv"

# Model load paths (supports primary container mount and absolute path fallback)
PRIMARY_MODEL_PATH = "/app/app/models/xgb_churn_model.pkl"
SECONDARY_MODEL_PATH = "/app/models/xgb_churn_model.pkl"

def get_model_path():
    """Helper to select active model binary file depending on volume mount setup."""
    if os.path.exists(PRIMARY_MODEL_PATH):
        return PRIMARY_MODEL_PATH
    if os.path.exists(SECONDARY_MODEL_PATH):
        return SECONDARY_MODEL_PATH
    # Fallback to current directory for local manual script debugging
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "models", "xgb_churn_model.pkl")
    return os.path.abspath(fallback)


# --- PIPELINE TASKS DEFINITIONS ---

def mock_raw_data_task():
    """
    Task 1: Create/mock raw Customer data for batch inference.
    Simulates extracting daily new account information from database staging tables.
    Saves results to the raw daily file.
    """
    logger.info("Initializing raw daily data extraction simulation...")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    # Generate mock customer accounts
    raw_df = pd.DataFrame([
        {
            "CustomerID": "1001-A", "tenure": 12, "MonthlyCharges": 70.05, "TotalCharges": 840.60,
            "Contract": "One year", "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
            "OnlineSecurity": 1, "OnlineBackup": 0, "DeviceProtection": 1, "TechSupport": 1,
            "StreamingTV": 0, "StreamingMovies": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0, "PaperlessBilling": 1
        },
        {
            "CustomerID": "2002-B", "tenure": 3, "MonthlyCharges": 45.15, "TotalCharges": 135.45,
            "Contract": "Month-to-month", "InternetService": "DSL", "PaymentMethod": "Mailed check",
            "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0,
            "StreamingTV": 0, "StreamingMovies": 1, "SeniorCitizen": 1, "Partner": 0, "Dependents": 0, "PaperlessBilling": 1
        },
        {
            "CustomerID": "3003-C", "tenure": 72, "MonthlyCharges": 115.80, "TotalCharges": 8337.60,
            "Contract": "Two year", "InternetService": "Fiber optic", "PaymentMethod": "Bank transfer (automatic)",
            "OnlineSecurity": 1, "OnlineBackup": 1, "DeviceProtection": 1, "TechSupport": 1,
            "StreamingTV": 1, "StreamingMovies": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 1, "PaperlessBilling": 0
        }
    ])

    raw_df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Staged {len(raw_df)} mock customer records successfully at: {RAW_DATA_PATH}")


def feature_engineering_task():
    """
    Task 2: Perform feature engineering.
    Calculates engineered feature 'total_services' by aggregating subscribed aux services.
    Saves to the engineered daily data file.
    """
    logger.info("Initializing Daily Feature Engineering Job...")
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Staged raw data not found at path: {RAW_DATA_PATH}")

    # Load staged raw csv
    df = pd.read_csv(RAW_DATA_PATH)

    # Columns representing auxiliary services to sum up
    service_columns = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    # Run engineering operation
    df["total_services"] = df[service_columns].sum(axis=1)
    
    # Save engineered staging file
    os.makedirs(os.path.dirname(ENGINEERED_DATA_PATH), exist_ok=True)
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    logger.info(f"Feature engineering successful. Staged dataset saved at: {ENGINEERED_DATA_PATH}")


def batch_inference_task():
    """
    Task 3: Batch Inference.
    Loads XGBoost model binary, preprocesses data, and outputs prediction report.
    """
    logger.info("Initializing XGBoost Daily Batch Prediction Pipeline...")

    if not os.path.exists(ENGINEERED_DATA_PATH):
        raise FileNotFoundError(f"Engineered dataset not found at: {ENGINEERED_DATA_PATH}")

    # Resolve active model path
    model_file = get_model_path()
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Pre-trained model pickle binary not found. Paths verified: {PRIMARY_MODEL_PATH} and {SECONDARY_MODEL_PATH}")

    logger.info(f"Loading pre-trained XGBoost Classifier from: {model_file}...")
    model = joblib.load(model_file)
    # Apply version-compatibility patch for model parameters
    import inspect
    params = set()
    for cls in inspect.getmro(model.__class__):
        if hasattr(cls, '_get_param_names'):
            params.update(cls._get_param_names())
    for p in params:
        if not hasattr(model, p):
            setattr(model, p, None)

    # Load engineered records
    df = pd.read_csv(ENGINEERED_DATA_PATH)

    # Pre-trained model expected columns schema in exact order
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

    # Preprocess categorical values and map dynamically to form the 24 column input matrix
    records = []
    for _, row in df.iterrows():
        # Init row vector with zero floats
        feat_row = {col: 0.0 for col in expected_features}
        
        # Populate numeric features directly
        feat_row['tenure'] = float(row['tenure'])
        feat_row['MonthlyCharges'] = float(row['MonthlyCharges'])
        feat_row['TotalCharges'] = float(row['TotalCharges'])
        feat_row['total_services'] = float(row['total_services'])
        feat_row['SeniorCitizen'] = float(row['SeniorCitizen'])
        feat_row['Partner'] = float(row['Partner'])
        feat_row['Dependents'] = float(row['Dependents'])
        feat_row['OnlineSecurity'] = float(row['OnlineSecurity'])
        feat_row['OnlineBackup'] = float(row['OnlineBackup'])
        feat_row['DeviceProtection'] = float(row['DeviceProtection'])
        feat_row['TechSupport'] = float(row['TechSupport'])
        feat_row['StreamingTV'] = float(row['StreamingTV'])
        feat_row['StreamingMovies'] = float(row['StreamingMovies'])
        feat_row['PaperlessBilling'] = float(row['PaperlessBilling'])

        # Dynamic One-Hot Encode mapping for 'Contract'
        contract_key = f"Contract_{row['Contract']}"
        if contract_key in feat_row:
            feat_row[contract_key] = 1.0

        # Dynamic One-Hot Encode mapping for 'InternetService'
        internet_key = f"InternetService_{row['InternetService']}"
        if internet_key in feat_row:
            feat_row[internet_key] = 1.0

        # Dynamic One-Hot Encode mapping for 'PaymentMethod'
        payment_key = f"PaymentMethod_{row['PaymentMethod']}"
        if payment_key in feat_row:
            feat_row[payment_key] = 1.0

        records.append(feat_row)

    # Construct input matrix
    input_df = pd.DataFrame(records)
    input_df = input_df[expected_features]
    
    # Cast to float to prevent training mismatch
    for col in expected_features:
        input_df[col] = input_df[col].astype(float)

    # Execute predictions on daily batch
    logger.info("Executing XGBoost batch predictions...")
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)[:, 1]

    # Build predictions outcome report dataframe
    results_df = pd.DataFrame({
        "CustomerID": df["CustomerID"],
        "churn_prediction": preds,
        "churn_probability": probs,
        "prediction_timestamp": datetime.now().isoformat()
    })

    # Save results to output directory
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Daily batch prediction report generated successfully at: {RESULTS_PATH}")


# --- AIRFLOW DAG ORCHESTRATION ---

with DAG(
    dag_id="telco_churn_daily_inference_pipeline",
    default_args=default_args,
    description="Automated daily customer churn batch prediction pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "batch"],
) as dag:

    # Task 1: stage raw daily records
    mock_data = PythonOperator(
        task_id="mock_raw_daily_data",
        python_callable=mock_raw_data_task
    )

    # Task 2: run feature engineering
    feature_engineering = PythonOperator(
        task_id="perform_feature_engineering",
        python_callable=feature_engineering_task
    )

    # Task 3: batch inference execution
    batch_predictions = PythonOperator(
        task_id="execute_batch_predictions",
        python_callable=batch_inference_task
    )

    # Pipeline task dependencies flow
    mock_data >> feature_engineering >> batch_predictions
