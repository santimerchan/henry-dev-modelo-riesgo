#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE="dspt06-riesgo-crediticio"
AWS_REGION="us-east-1"
ECR_REPO="dspt06-riesgo-crediticio"
LAMBDA_FUNCTION="dspt06-riesgo-crediticio-predict"
LAMBDA_ROLE="dspt06-riesgo-crediticio-lambda-role"
API_NAME="dspt06-riesgo-crediticio-api"

ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

# 1. Rol IAM para Lambda (crear si no existe)
if ! aws iam get-role --role-name "$LAMBDA_ROLE" --profile "$AWS_PROFILE" >/dev/null 2>&1; then
  echo "Creando rol IAM $LAMBDA_ROLE..."
  cat > lambda-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]
}
EOF
  aws iam create-role --role-name "$LAMBDA_ROLE" \
    --assume-role-policy-document file://lambda-trust-policy.json --profile "$AWS_PROFILE"
  aws iam attach-role-policy --role-name "$LAMBDA_ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole --profile "$AWS_PROFILE"
  echo "Esperando propagación del rol (10s)..."
  sleep 10
else
  echo "Rol $LAMBDA_ROLE ya existe, se reutiliza."
fi

# 2. Repositorio ECR (crear si no existe)
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Creando repositorio ECR $ECR_REPO..."
  aws ecr create-repository --repository-name "$ECR_REPO" --profile "$AWS_PROFILE" --region "$AWS_REGION"
else
  echo "Repositorio $ECR_REPO ya existe, se reutiliza."
fi

# 3. Build y push de la imagen
docker build -t "$ECR_REPO:latest" .
aws ecr get-login-password --region "$AWS_REGION" --profile "$AWS_PROFILE" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker tag "$ECR_REPO:latest" "$IMAGE_URI"
docker push "$IMAGE_URI"

# 4. Función Lambda (crear o actualizar)
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE}"
if aws lambda get-function --function-name "$LAMBDA_FUNCTION" --profile "$AWS_PROFILE" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Función $LAMBDA_FUNCTION ya existe, actualizando imagen..."
  aws lambda update-function-code --function-name "$LAMBDA_FUNCTION" --image-uri "$IMAGE_URI" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION"
else
  echo "Creando función $LAMBDA_FUNCTION..."
  aws lambda create-function \
    --function-name "$LAMBDA_FUNCTION" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$LAMBDA_ROLE_ARN" \
    --timeout 30 --memory-size 512 \
    --profile "$AWS_PROFILE" --region "$AWS_REGION"
fi

LAMBDA_ARN=$(aws lambda get-function --function-name "$LAMBDA_FUNCTION" \
  --profile "$AWS_PROFILE" --region "$AWS_REGION" --query "Configuration.FunctionArn" --output text)

# 5. API Gateway HTTP API (crear si no existe)
API_ID=$(aws apigatewayv2 get-apis --profile "$AWS_PROFILE" --region "$AWS_REGION" \
  --query "Items[?Name=='${API_NAME}'].ApiId" --output text)

if [ -z "$API_ID" ]; then
  echo "Creando API Gateway $API_NAME..."
  API_ID=$(aws apigatewayv2 create-api \
    --name "$API_NAME" --protocol-type HTTP --target "$LAMBDA_ARN" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" --query "ApiId" --output text)
else
  echo "API Gateway $API_NAME ya existe (ApiId=$API_ID), se reutiliza."
fi

# 6. Permiso para que API Gateway invoque el Lambda (se ignora si ya existía)
aws lambda add-permission \
  --function-name "$LAMBDA_FUNCTION" \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:${AWS_REGION}:${ACCOUNT_ID}:${API_ID}/*/*" \
  --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || echo "Permiso ya existía, se ignora."

API_URL="https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com"
echo ""
echo "Listo. Endpoint disponible en: ${API_URL}/predict"
