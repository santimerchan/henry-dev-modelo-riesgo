"""
Feature engineering para el modelo de riesgo crediticio (Avance 2).
Preprocesamiento reutilizable: imputación, codificación y split train/test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Rutas y parámetros del experimento
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "Base_de_datos.csv"
TARGET_COLUMN = "Pago_atiempo"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columnas elegidas en comprension_eda.ipynb (sección 10 — hallazgos)
numeric_features = [
    "edad_cliente",
    "capital_prestado",
    "plazo_meses",
    "puntaje_datacredito",
    "huella_consulta",
]

categoric_features = ["tipo_laboral"]

# tipo_credito es código de producto; OrdinalEncoder respeta el orden del código
categoric_ordinal_features = ["tipo_credito"]
ordinal_categories = [[4, 6, 7, 9, 10, 68]]


def load_data(data_path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Lee Base_de_datos.csv desde la raíz del repositorio."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {data_path}. "
            "Verifica que `Base_de_datos.csv` esté en la raíz del repositorio."
        )
    return pd.read_csv(data_path)


def build_feature_pipeline(
    numeric_features: list[str],
    categoric_features: list[str],
    categoric_ordinal_features: list[str],
    ordinal_categories: list[list] | None = None,
) -> Pipeline:
    """Arma el ColumnTransformer con tres ramas: numérica, nominal y ordinal."""
    if ordinal_categories is None:
        ordinal_categories = [[] for _ in categoric_ordinal_features]

    if len(categoric_ordinal_features) != len(ordinal_categories):
        raise ValueError(
            "categoric_ordinal_features y ordinal_categories deben tener la misma longitud."
        )

    # Numéricas: imputar con mediana (robusta ante outliers de montos y edad)
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # Nominales: imputar con moda y expandir con one-hot
    categoric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Ordinales: imputar con moda y codificar respetando el orden definido
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categoric_features:
        transformers.append(("categoric", categoric_transformer, categoric_features))
    if categoric_ordinal_features:
        transformers.append(
            ("categoric_ordinales", ordinal_transformer, categoric_ordinal_features)
        )

    if not transformers:
        raise ValueError("Debes definir al menos una columna en las listas de features.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa predictores y target; estratifica por Pago_atiempo (~95% / ~5%)."""
    if target_column not in df.columns:
        raise KeyError(f"La columna objetivo '{target_column}' no existe en el dataset.")

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"Columnas no encontradas en el dataset: {missing_features}")

    X = df[feature_columns]
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_transform_datasets(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta solo con train; transforma test sin volver a aprender (evita data leakage)."""
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    return X_train_transformed, X_test_transformed


def get_all_feature_columns(
    numeric_features: list[str],
    categoric_features: list[str],
    categoric_ordinal_features: list[str],
) -> list[str]:
    """Concatena las tres listas de columnas predictoras."""
    return numeric_features + categoric_features + categoric_ordinal_features


if __name__ == "__main__":
    print("=== Feature engineering ===")
    df = load_data()
    print(f"Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

    feature_columns = get_all_feature_columns(
        numeric_features, categoric_features, categoric_ordinal_features
    )

    pipeline = build_feature_pipeline(
        numeric_features=numeric_features,
        categoric_features=categoric_features,
        categoric_ordinal_features=categoric_ordinal_features,
        ordinal_categories=ordinal_categories,
    )

    X_train, X_test, y_train, y_test = split_features_target(
        df=df,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
    )

    X_train_transformed, X_test_transformed = fit_transform_datasets(
        pipeline, X_train, X_test
    )

    print(f"X_train_transformed: {X_train_transformed.shape}")
    print(f"X_test_transformed:  {X_test_transformed.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")
