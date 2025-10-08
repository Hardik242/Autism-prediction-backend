from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

from app_utils import (
    load_all_models_and_artifacts,
    preprocess_input,
    get_feature_importance_explanation,
)

# --- Application Setup ---
app = FastAPI(
    title="Specialized ASD Screening API",
    description="An API with specialized models for toddlers and the general population.",
    version="2.2.0",  # Version bump
)
app_data = {}

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
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


# --- API Events ---
@app.on_event("startup")
def startup_event():
    global app_data
    app_data = load_all_models_and_artifacts()


# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Specialized ASD Screening API."}


@app.post("/predict/", response_model=List[PredictionResult], tags=["Prediction"])
def predict(data: ScreeningData):
    """
    Routes the prediction to the correct set of models based on the 'isToddler' flag.
    Always returns a list of predictions from the top 3 models for that category.
    """
    if not app_data:
        raise HTTPException(status_code=503, detail="Models are not available.")

    user_input_dict = data.dict()
    is_toddler = user_input_dict.pop("isToddler")

    model_group = "toddler" if is_toddler else "general"
    model_prefix = "Toddler Specialist" if is_toddler else "General"

    artifacts = app_data[model_group]["artifacts"]
    models_to_run = app_data[model_group]["models"]

    # --- Validation ---
    if is_toddler:
        max_age = artifacts["max_toddler_age_months"]
        if data.Age > max_age:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid age for toddler. Max age is {max_age} months.",
            )

    # --- Preprocessing & Prediction Loop ---
    try:
        processed_df = preprocess_input(user_input_dict, artifacts, is_toddler)
        results = []

        for name, model in models_to_run.items():
            proba = model.predict_proba(processed_df)[0]
            explanation = get_feature_importance_explanation(
                model, artifacts["preprocessors"]["feature_order"]
            )

            # Reformat model name for clarity
            clean_name = name.split(" ", 1)[1]  # "1 Adaboost" -> "Adaboost"

            results.append(
                PredictionResult(
                    model_name=f"{model_prefix} ({clean_name})",
                    risk_percent=proba[1] * 100,
                    explanation=explanation,
                )
            )

        if not results:
            raise ValueError("No predictions were generated.")

        return sorted(
            results, key=lambda x: x.model_name
        )  # Sort by name for consistent order

    except Exception as e:
        # Provide a more specific error message for debugging
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}"
        )
