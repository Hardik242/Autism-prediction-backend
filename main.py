from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from app_utils import (
    load_all_models_and_artifacts,
    preprocess_input,
    get_feature_importance_explanation,
)

app = FastAPI(
    title="Specialized ASD Screening API",
    description="An API with specialized models for toddlers and the general population.",
    version="2.2.0",
)
app_data = {}

origins = [
    "http://localhost:8080",  # Your local frontend development server
    "https://autism-prediction-frontend.vercel.app",  # Your deployed frontend URL
]

# Allow the frontend (running on a different origin) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use the specific list of origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ScreeningData(BaseModel):
    isToddler: bool
    Age: int = Field(..., gt=0)
    Gender: str
    Jaundice: str
    Family_History_ASD: str
    A1_Score: str
    A2_Score: str
    A3_Score: str
    A4_Score: str
    A5_Score: str
    A6_Score: str
    A7_Score: str
    A8_Score: str
    A9_Score: str
    A10_Score: str


class PredictionResult(BaseModel):
    model_name: str
    risk_percent: float
    explanation: List[str]


@app.on_event("startup")
def startup_event():
    """
    Loads all machine learning assets into memory once when the server starts.
    """
    global app_data
    app_data = load_all_models_and_artifacts()


@app.get("/", tags=["General"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Specialized ASD Screening API."}


@app.post("/predict/", response_model=List[PredictionResult], tags=["Prediction"])
def predict(data: ScreeningData):
    """
    Routes the prediction to the correct set of models based on the 'isToddler' flag
    and returns a list of predictions.
    """
    if not app_data:
        raise HTTPException(status_code=503, detail="Models are not available.")

    user_input_dict = data.dict()
    is_toddler = user_input_dict.pop("isToddler")

    model_group = "toddler" if is_toddler else "general"
    model_prefix = "Toddler Specialist" if is_toddler else "General"

    artifacts = app_data[model_group]["artifacts"]
    models_to_run = app_data[model_group]["models"]

    if is_toddler:
        max_age = artifacts["max_toddler_age_months"]
        if data.Age > max_age:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid age for toddler. Max age is {max_age} months.",
            )

    try:
        processed_df = preprocess_input(user_input_dict, artifacts, is_toddler)
        results = []

        for name, model in models_to_run.items():
            proba = model.predict_proba(processed_df)[0]
            explanation = get_feature_importance_explanation(
                model, artifacts["preprocessors"]["feature_order"]
            )

            clean_name = name.split(" ", 1)[1]

            results.append(
                PredictionResult(
                    model_name=f"{model_prefix} ({clean_name})",
                    risk_percent=proba[1] * 100,
                    explanation=explanation,
                )
            )

        if not results:
            raise ValueError("No predictions were generated.")

        # Sort by name to ensure a consistent order in the API response.
        return sorted(results, key=lambda x: x.model_name)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}"
        )
