from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .nslkdd import FEATURE_NAMES, CATEGORICAL, NUMERIC

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC),
        ("cat", categorical_pipeline, CATEGORICAL),
    ])

def fit_transform(train_df, model_dir="models/nslkdd"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(train_df[FEATURE_NAMES])
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    return X, preprocessor

def transform(df, model_dir="models/nslkdd"):
    model_dir = Path(model_dir)
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    return preprocessor.transform(df[FEATURE_NAMES])
