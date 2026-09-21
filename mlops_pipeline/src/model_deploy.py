"""
Disponibilización del modelo de riesgo crediticio como API HTTP (Avance 4).

Expone el modelo entrenado por `model_training_evaluation.py` en un endpoint `/predict`,
para que cualquier sistema pueda pedir una predicción sin saber Python ni tener el dataset.

El archivo `modelo_riesgo.joblib` ya trae dentro el preprocesamiento de `ft_engineering.py`
ajustado durante el entrenamiento, así que aquí no se repite ninguna transformación: se
reutiliza la misma, que es la única forma de que el modelo reciba los datos como los aprendió.

Uso local (desde la raíz del repo):
    uvicorn mlops_pipeline.src.model_deploy:app --reload --port 8000
    # documentación interactiva en http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

# Los módulos del pipeline se importan de forma plana (`from ft_engineering import ...`),
# igual que en model_training_evaluation.py. Agregar esta carpeta al path permite arrancar
# uvicorn desde cualquier directorio, incluida la raíz del contenedor.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ft_engineering import (  # noqa: E402  (requiere el sys.path de arriba)
    REPO_ROOT,
    TARGET_COLUMN,
    categoric_features,
    categoric_ordinal_features,
    get_all_feature_columns,
    numeric_features,
)

# Debe coincidir con MODEL_FILENAME de model_training_evaluation.py, que es quien lo escribe.
# No se importa desde ahí para no arrastrar al servicio las dependencias de entrenamiento
# (matplotlib, seaborn, MLflow) y mantener liviano el arranque de la API.
MODEL_FILENAME = "modelo_riesgo.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", REPO_ROOT / MODEL_FILENAME))

# Columnas crudas que espera el modelo, en el mismo orden con el que se entrenó.
FEATURE_COLUMNS = get_all_feature_columns(
    numeric_features, categoric_features, categoric_ordinal_features
)

# Esquema del registro para la documentación de /docs. Los tipos salen de Base_de_datos.csv.
RECORD_SCHEMA = {
    "type": "object",
    "required": FEATURE_COLUMNS,
    "properties": {
        "edad_cliente": {"type": "integer", "example": 42},
        "capital_prestado": {"type": "number", "example": 3692160.0},
        "plazo_meses": {"type": "integer", "example": 10},
        "puntaje_datacredito": {"type": "number", "example": 695.0},
        "huella_consulta": {"type": "integer", "example": 5},
        "tipo_laboral": {
            "type": "string",
            "enum": ["Independiente", "Empleado"],
            "example": "Independiente",
        },
        "tipo_credito": {"type": "integer", "enum": [4, 6, 7, 9, 10, 68], "example": 7},
    },
}

# /predict recibe dos formatos distintos, así que el cuerpo se describe a mano para que
# Swagger UI ofrezca las dos opciones: pegar JSON o subir un CSV.
PREDICT_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "oneOf": [
                        RECORD_SCHEMA,
                        {"type": "array", "items": RECORD_SCHEMA, "minItems": 1},
                    ],
                    "description": "Un registro único o una lista de registros (batch).",
                }
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "CSV con una fila por solicitud a evaluar.",
                        }
                    },
                    "required": ["file"],
                }
            },
        },
    }
}


def load_serving_pipeline(model_path: Path | str = MODEL_PATH):
    """
    Carga el pipeline (preprocesamiento + modelo) serializado con joblib.

    Se llama una sola vez al importar el módulo: leer el archivo de disco es lento
    comparado con predecir, y no tiene sentido repetirlo en cada solicitud.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {model_path}. "
            "Corre `python mlops_pipeline/src/model_training_evaluation.py` para generarlo."
        )
    return joblib.load(model_path)


model = load_serving_pipeline()
MODEL_NAME = type(model.named_steps["model"]).__name__

app = FastAPI(
    title="API de riesgo crediticio",
    description=(
        "Predice si un cliente pagará a tiempo (`Pago_atiempo`). "
        "Acepta JSON (un registro o una lista) y archivos CSV."
    ),
    version="1.0.0",
)


def dataframe_from_json(payload) -> pd.DataFrame:
    """Convierte el JSON recibido en DataFrame; acepta un objeto o una lista de objetos."""
    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list) or not payload:
        raise HTTPException(
            status_code=422,
            detail="El cuerpo debe ser un objeto JSON o una lista de objetos no vacía.",
        )

    if not all(isinstance(item, dict) for item in payload):
        raise HTTPException(
            status_code=422, detail="Cada elemento de la lista debe ser un objeto JSON."
        )

    return pd.DataFrame(payload)


def dataframe_from_csv(contenido: bytes) -> pd.DataFrame:
    """Lee el CSV subido como multipart/form-data."""
    try:
        return pd.read_csv(io.BytesIO(contenido))
    except Exception as error:
        raise HTTPException(
            status_code=422, detail=f"No se pudo leer el CSV: {error}"
        ) from error


def validar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confirma que lleguen todas las columnas predictoras y las devuelve en el orden esperado.

    Las columnas de más se descartan: así un CSV exportado del dataset completo (que trae
    también el target y variables no usadas) funciona sin tener que recortarlo a mano.
    """
    if df.empty:
        raise HTTPException(status_code=422, detail="No se recibió ningún registro.")

    faltantes = [columna for columna in FEATURE_COLUMNS if columna not in df.columns]
    if faltantes:
        raise HTTPException(
            status_code=422,
            detail=f"Faltan columnas requeridas: {faltantes}",
        )

    return df[FEATURE_COLUMNS]


def predecir(df: pd.DataFrame) -> list[dict]:
    """Aplica el pipeline completo y arma una respuesta por registro."""
    try:
        etiquetas = model.predict(df)
        probabilidades = model.predict_proba(df)[:, 1]
    except Exception as error:
        raise HTTPException(
            status_code=422, detail=f"El modelo no pudo procesar los datos: {error}"
        ) from error

    return [
        {
            "indice": indice,
            "prediccion": int(etiqueta),
            "probabilidad_pago_atiempo": round(float(probabilidad), 4),
            "probabilidad_incumplimiento": round(1 - float(probabilidad), 4),
        }
        for indice, (etiqueta, probabilidad) in enumerate(zip(etiquetas, probabilidades))
    ]


@app.get("/")
def raiz() -> dict:
    """Datos básicos del servicio, útiles para confirmar que la imagen quedó bien armada."""
    return {
        "servicio": "API de riesgo crediticio",
        "modelo": MODEL_NAME,
        "target": TARGET_COLUMN,
        "columnas_esperadas": FEATURE_COLUMNS,
        "documentacion": "/docs",
    }


@app.get("/health")
def health() -> dict:
    """Chequeo de salud: responde 200 solo si el modelo quedó cargado en memoria."""
    return {"status": "ok", "modelo_cargado": model is not None}


@app.post("/predict", openapi_extra=PREDICT_REQUEST_BODY)
async def predict(request: Request) -> dict:
    """
    Predice `Pago_atiempo` para uno o varios clientes.

    Acepta dos formatos, ambos con soporte de batch:
    - `application/json`: un objeto con las columnas del cliente, o una lista de objetos.
    - `multipart/form-data`: un archivo CSV con una fila por cliente.

    `prediccion` es 1 si el modelo espera pago a tiempo y 0 si espera incumplimiento;
    `probabilidad_pago_atiempo` es la probabilidad de la clase 1.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        formulario = await request.form()
        archivos = [valor for valor in formulario.values() if hasattr(valor, "read")]
        if not archivos:
            raise HTTPException(
                status_code=422,
                detail="Envía el CSV como archivo, por ejemplo: -F 'file=@clientes.csv'.",
            )
        df = dataframe_from_csv(await archivos[0].read())
    else:
        try:
            payload = await request.json()
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail="Cuerpo inválido: se esperaba JSON o un CSV en multipart/form-data.",
            ) from error
        df = dataframe_from_json(payload)

    predicciones = predecir(validar_columnas(df))

    return {
        "modelo": MODEL_NAME,
        "n_registros": len(predicciones),
        "predicciones": predicciones,
    }


if __name__ == "__main__":
    import uvicorn

    print(f"Modelo: {MODEL_NAME} | artefacto: {MODEL_PATH}")
    print(f"Raíz del repositorio: {REPO_ROOT}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
