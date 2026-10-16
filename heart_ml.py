from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path('.')
DATA_PATH = BASE_DIR / 'data' / 'heart.csv'
MODEL_PATH = BASE_DIR / 'model.joblib'
METRICS_PATH = BASE_DIR / 'metrics.json'

NUMERIC = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'target'  # 0 = sin enfermedad, 1 = con enfermedad


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    from ucimlrepo import fetch_ucirepo
    heart = fetch_ucirepo(id=45)
    df = heart.data.features.copy()
    df[TARGET] = (heart.data.targets['num'] > 0).astype(int)
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', numeric_pipe, NUMERIC), ('cat', categorical_pipe, CATEGORICAL)])
