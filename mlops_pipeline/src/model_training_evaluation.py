"""
Entrenamiento y evaluación de modelos de clasificación (Avance 2).
Importa el preprocesamiento de ft_engineering.py para comparar modelos en las mismas condiciones.

Cada modelo candidato se entrena y evalúa dentro de un run de MLflow (tracking local en
./mlruns, junto a este script). Eso permite comparar hiperparámetros/métricas en la UI de
MLflow y recuperar el mejor modelo entrenado (por run_id) sin tener que volver a correr
el script — por ejemplo desde el futuro model_deploy.py.
"""

from __future__ import annotations

import os

# MLflow 3.x exige base de datos por defecto; este proyecto usa tracking local en ./mlruns
# (mismo patrón que HO1). Sin esta variable, `set_experiment` falla en clase.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ft_engineering import (
    RANDOM_STATE,
    REPO_ROOT,
    TARGET_COLUMN,
    build_feature_pipeline,
    categoric_features,
    categoric_ordinal_features,
    fit_transform_datasets,
    get_all_feature_columns,
    load_data,
    numeric_features,
    ordinal_categories,
    split_features_target,
)

# Tracking local de MLflow: vive junto a este script (mismo patrón que HO1), resuelto por
# ruta absoluta para que el resultado no dependa del directorio desde el que se ejecute.
MLRUNS_DIR = Path(__file__).resolve().parent / "mlruns"
EXPERIMENT_NAME = "modelo_riesgo_crediticio"

# Artefacto que consume model_deploy.py. MLflow guarda cada candidato para poder comparar
# experimentos, pero `mlruns/` está en .gitignore y no viaja en la imagen Docker: el modelo
# ganador se serializa además como un archivo propio en la raíz del repositorio.
# Se usa joblib (no pickle puro) porque los modelos de sklearn guardan internamente arreglos
# grandes de NumPy y joblib los serializa mejor.
MODEL_FILENAME = "modelo_riesgo.joblib"
MODEL_PATH = REPO_ROOT / MODEL_FILENAME


def build_model(model, X_train, y_train, sample_weight=None):
    """Entrena cualquier estimador de sklearn (o compatible, como XGBoost) y lo retorna ajustado."""
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def summarize_classification(model, X_test, y_test, model_name: str) -> dict:
    """
    Métricas de clasificación:
    - accuracy: aciertos totales (puede engañar con clases desbalanceadas).
    - precision: de los predichos positivos, cuántos lo eran.
    - recall: de los positivos reales, cuántos detectamos.
    - f1: balance entre precision y recall.
    - roc_auc: capacidad de separar clases en distintos umbrales.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def train_candidate_models(X_train, y_train) -> dict:
    """
    Entrena siete modelos candidatos: los tres originales (lineal, bosque aleatorio,
    gradient boosting) más ExtraTrees, SVC, KNN y XGBoost.

    El target está desbalanceado (~95% / 5% en Pago_atiempo), así que cada modelo compensa
    las clases con el mecanismo que soporta:
    - `class_weight='balanced'`: LogisticRegression, RandomForest, ExtraTrees, SVC.
    - `sample_weight` balanceado pasado a `fit()`: GradientBoosting (no acepta `class_weight`).
    - `scale_pos_weight`: XGBoost.
    - KNeighbors no soporta ponderar clases (es un modelo basado en distancia); queda sin
      compensar y es esperable que su recall en la clase minoritaria sea más bajo.
    """
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count
    balanced_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # name -> (estimador, sample_weight a pasar en fit() o None)
    candidates = {
        "LogisticRegression": (
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
            None,
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            None,
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                class_weight="balanced",
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            None,
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            balanced_sample_weight,
        ),
        "SVC": (
            SVC(
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
            None,
        ),
        "KNeighbors": (KNeighborsClassifier(), None),
        "XGBoost": (
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            None,
        ),
    }

    return {
        name: build_model(estimator, X_train, y_train, sample_weight=sample_weight)
        for name, (estimator, sample_weight) in candidates.items()
    }


def build_summary_table(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Tabla comparativa: una fila por modelo, columnas por métrica."""
    rows = [
        summarize_classification(model, X_test, y_test, model_name)
        for model_name, model in trained_models.items()
    ]
    return pd.DataFrame(rows).set_index("model_name")


def get_loggable_params(model) -> dict:
    """Hiperparámetros del estimador filtrados a tipos simples (MLflow no acepta objetos)."""
    return {
        key: value
        for key, value in model.get_params().items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def log_model_run(model, model_name: str, metrics: dict, X_test_sample) -> str:
    """
    Abre un run anidado de MLflow para un modelo candidato y loguea:
    - hiperparámetros (`get_params()`),
    - métricas de `summarize_classification`,
    - el modelo entrenado como artefacto (flavor xgboost o sklearn según corresponda).

    Retorna el `run_id`, para poder recuperar el mejor modelo después con
    `mlflow.pyfunc.load_model(f"runs:/{run_id}/model")`.
    """
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        mlflow.log_params(get_loggable_params(model))
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model_name"})

        log_model_fn = (
            mlflow.xgboost.log_model
            if isinstance(model, XGBClassifier)
            else mlflow.sklearn.log_model
        )
        log_model_fn(model, artifact_path="model", input_example=X_test_sample[:5])

        return run.info.run_id


def train_and_log_models(X_train, y_train, X_test, y_test) -> tuple[dict, pd.DataFrame, dict]:
    """
    Entrena todos los modelos candidatos y loguea cada uno como un run anidado de MLflow
    dentro del run activo (debe llamarse dentro de un `with mlflow.start_run(...)`).

    Retorna (trained_models, summary_df, run_ids), donde `run_ids` mapea model_name -> run_id.
    """
    trained_models = train_candidate_models(X_train, y_train)

    summary_rows = []
    run_ids = {}
    for model_name, model in trained_models.items():
        metrics = summarize_classification(model, X_test, y_test, model_name)
        summary_rows.append(metrics)
        run_ids[model_name] = log_model_run(model, model_name, metrics, X_test)

    summary_df = pd.DataFrame(summary_rows).set_index("model_name")
    return trained_models, summary_df, run_ids


def plot_metric_comparison(summary_df: pd.DataFrame) -> None:
    """Barras comparando F1 y ROC-AUC entre modelos."""
    metrics_to_plot = summary_df[["f1", "roc_auc"]].reset_index()
    melted = metrics_to_plot.melt(
        id_vars="model_name", var_name="metrica", value_name="valor"
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="model_name", y="valor", hue="metrica")
    plt.title("Comparación de modelos — F1 y ROC-AUC")
    plt.xlabel("Modelo")
    plt.ylabel("Valor de la métrica")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    plt.tight_layout()


def plot_roc_curves(trained_models: dict, X_test, y_test) -> None:
    """Curvas ROC superpuestas para ver separación de clases por umbral."""
    plt.figure(figsize=(8, 6))

    for model_name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    plt.title("Curvas ROC superpuestas")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend()
    plt.tight_layout()


def select_best_model(summary_df: pd.DataFrame, metric: str = "roc_auc") -> str:
    """
    Elige el modelo con mejor ROC-AUC en test.
    En producción también importan consistencia temporal y escalabilidad, no solo la métrica.
    """
    return summary_df[metric].idxmax()


def build_serving_pipeline(feature_pipeline: Pipeline, model) -> Pipeline:
    """
    Une el preprocesador ya ajustado y el modelo ganador en un solo objeto.

    Ambos llegan entrenados, así que no se vuelve a llamar `fit()`: el resultado acepta
    directamente un DataFrame con las columnas crudas y devuelve la predicción.
    """
    return Pipeline(steps=[("preprocessor", feature_pipeline), ("model", model)])


def save_best_model(
    feature_pipeline: Pipeline,
    model,
    model_path: Path | str = MODEL_PATH,
) -> Path:
    """
    Serializa preprocesamiento + modelo como un único archivo .joblib.

    Guardar los dos juntos evita el error clásico de servir el modelo con transformaciones
    distintas a las del entrenamiento: quien cargue el archivo recibe el pipeline completo.
    """
    model_path = Path(model_path)
    joblib.dump(build_serving_pipeline(feature_pipeline, model), model_path)
    return model_path


if __name__ == "__main__":
    print("=== Entrenamiento y evaluación ===")

    df = load_data()
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

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="comparacion_modelos") as parent_run:
        mlflow.log_params(
            {
                "n_train": X_train_transformed.shape[0],
                "n_test": X_test_transformed.shape[0],
                "n_features": X_train_transformed.shape[1],
                "target_column": TARGET_COLUMN,
                "random_state": RANDOM_STATE,
            }
        )

        trained_models, summary_df, run_ids = train_and_log_models(
            X_train_transformed, y_train, X_test_transformed, y_test
        )

        print("\n--- Tabla resumen de métricas ---")
        print(summary_df.round(4))

        best_model_name = select_best_model(summary_df, metric="roc_auc")
        best_run_id = run_ids[best_model_name]

        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_metric("best_model_roc_auc", summary_df.loc[best_model_name, "roc_auc"])

        saved_path = save_best_model(pipeline, trained_models[best_model_name])
        print(f"Modelo serializado en: {saved_path}")

        print(f"\nMejor modelo según ROC-AUC: {best_model_name} (run_id={best_run_id})")
        print(f"Experimento MLflow: {EXPERIMENT_NAME} | tracking_uri: {MLRUNS_DIR.as_uri()}")
        print(
            "Para cargarlo en otro script (p.ej. model_deploy.py):\n"
            f'  mlflow.set_tracking_uri("{MLRUNS_DIR.as_uri()}")\n'
            f'  model = mlflow.pyfunc.load_model("runs:/{best_run_id}/model")'
        )

    plot_metric_comparison(summary_df)
    plot_roc_curves(trained_models, X_test_transformed, y_test)
    plt.show()
