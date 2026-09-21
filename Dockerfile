FROM python:3.11-slim

WORKDIR /app

# Las dependencias se instalan antes de copiar el código: mientras requirements.txt no
# cambie, Docker reutiliza esta capa en cache y los builds siguientes son mucho más rápidos.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código del pipeline y modelo ya entrenado (generado por model_training_evaluation.py).
COPY mlops_pipeline/ ./mlops_pipeline/
COPY modelo_riesgo.joblib ./

EXPOSE 8000

CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]

# Permite que esta misma imagen corra también en AWS Lambda, sin tocar el código
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8000
