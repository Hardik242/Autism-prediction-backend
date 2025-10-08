import os
import joblib
import pandas as pd
import numpy as np

# --- Constants ---
TODDLER_MODELS_DIR = "models/toddler"
GENERAL_MODELS_DIR = "models/general"

# --- Unified Answer and Scoring Logic ---
SCORE_MAPPING = {
    "aq_agree_positive": ["Always", "Usually"],
    "aq_disagree_positive": ["Rarely", "Never"],
    "toddler_freq_positive": ["Always", "Usually", "Sometimes"],
    "toddler_freq_negative": ["Rarely", "Never"],
}


def load_all_models_and_artifacts():
    """
    Loads the top 3 toddler and general models and their artifacts at startup.
    """
    app_data = {"toddler": {"models": {}}, "general": {"models": {}}}
    try:
        # Load Top 3 Toddler Models & Artifacts
        for filename in os.listdir(TODDLER_MODELS_DIR):
            if filename.startswith("toddler_model_rank_") and filename.endswith(
                ".joblib"
            ):
                model_name = (
                    filename.replace("toddler_model_rank_", "")
                    .replace(".joblib", "")
                    .replace("_", " ")
                    .title()
                )
                model_path = os.path.join(TODDLER_MODELS_DIR, filename)
                app_data["toddler"]["models"][model_name] = joblib.load(model_path)
        app_data["toddler"]["artifacts"] = joblib.load(
            os.path.join(TODDLER_MODELS_DIR, "toddler_artifacts.joblib")
        )
        print(
            f"✅ {len(app_data['toddler']['models'])} Toddler models and artifacts loaded."
        )

        # Load Top 3 General Models & Artifacts
        for filename in os.listdir(GENERAL_MODELS_DIR):
            if filename.startswith("general_model_rank_") and filename.endswith(
                ".joblib"
            ):
                model_name = (
                    filename.replace("general_model_rank_", "")
                    .replace(".joblib", "")
                    .replace("_", " ")
                    .title()
                )
                model_path = os.path.join(GENERAL_MODELS_DIR, filename)
                app_data["general"]["models"][model_name] = joblib.load(model_path)
        app_data["general"]["artifacts"] = joblib.load(
            os.path.join(GENERAL_MODELS_DIR, "general_artifacts.joblib")
        )
        print(
            f"✅ {len(app_data['general']['models'])} General models and artifacts loaded."
        )

        return app_data
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None


def convert_user_input_to_scores(user_input, is_toddler):
    scores = {}
    for i in range(1, 11):
        key = f"A{i}_Score"
        answer = user_input.get(key)
        score = 0
        if is_toddler:
            if i in [7, 9]:
                if answer in SCORE_MAPPING["toddler_freq_negative"]:
                    score = 1
            else:
                if answer in SCORE_MAPPING["toddler_freq_positive"]:
                    score = 1
        else:
            if i in [1, 7, 8, 10]:
                if answer in SCORE_MAPPING["aq_agree_positive"]:
                    score = 1
            else:
                if answer in SCORE_MAPPING["aq_disagree_positive"]:
                    score = 1
        scores[key] = score
    return scores


def preprocess_input(user_input_dict, artifacts, is_toddler):
    if is_toddler:
        user_input_dict["Age_Mons"] = user_input_dict.pop("Age")

    df = pd.DataFrame([user_input_dict])
    scores = convert_user_input_to_scores(user_input_dict, is_toddler)
    for key, value in scores.items():
        df[key] = value

    df = df[artifacts["preprocessors"]["feature_order"]]

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numerical_cols:
        df[col] = df[col].astype(float)
    df.loc[:, numerical_cols] = artifacts["preprocessors"]["imputers"][
        "numerical"
    ].transform(df[numerical_cols])

    for col in categorical_cols:
        df.loc[:, col] = (
            artifacts["preprocessors"]["imputers"][col].transform(df[[col]]).ravel()
        )
        le = artifacts["preprocessors"]["encoders"][col]
        df[col] = df[col].apply(
            lambda x: x if str(x) in le.classes_ else le.classes_[0]
        )
        df[col] = le.transform(df[col])
        df[col] = df[col].astype(int)

    df.loc[:, numerical_cols] = artifacts["preprocessors"]["scaler"].transform(
        df[numerical_cols]
    )
    return df


def get_feature_importance_explanation(model, feature_names):
    if hasattr(model, "feature_importances_"):
        indices = np.argsort(model.feature_importances_)[::-1]
        return [feature_names[i] for i in indices[:5]]
    elif hasattr(model, "coef_"):
        indices = np.argsort(np.abs(model.coef_[0]))[::-1]
        return [feature_names[i] for i in indices[:5]]
    else:
        return ["Explanation not available for this model type."]
