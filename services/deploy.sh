#!/bin/bash
set -e

echo "🚀 BigQuery Streamlit AutoML - Backend Deployment"
echo "=================================================="

# === 1. LOAD ENVIRONMENT CONFIGURATION ===
echo ""
echo "📋 Loading Configuration"
echo "------------------------"

ENV_PATH="../.env"
if [ ! -f "$ENV_PATH" ]; then
  echo "❌ Error: .env file not found in project root"
  echo "   Please run setup.sh first from the project root directory"
  exit 1
fi

echo "✅ Loading configuration from .env..."
source "$ENV_PATH"

# Validate required environment variables
REQUIRED_VARS=(
    "PROJECT_ID"
    "REGION" 
    "REGION2"
    "SERVICE_ACCOUNT_EMAIL"
    "DATASET_ID"
    "BUCKET_NAME"
    "FUNCTION_NAME"
    "TRAINING_SERVICE"
    "PREDICT_SERVICE"
    "ARTIFACT_REPO"
)

echo "🔍 Validating configuration..."
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "❌ Error: $VAR is not set in .env file"
        exit 1
    fi
done

echo "✅ Configuration validated successfully"
echo ""
echo "📊 Deployment Configuration:"
echo "  • Project ID: $PROJECT_ID"
echo "  • Primary Region: $REGION"
echo "  • Secondary Region: $REGION2"
echo "  • Dataset: $DATASET_ID"
echo "  • Bucket: $BUCKET_NAME"
echo "  • Service Account: $SERVICE_ACCOUNT_EMAIL"

# === 2. SET GCP CONFIGURATION ===
echo ""
echo "🔧 Setting GCP Configuration"
echo "---------------------------"

gcloud config set project "$PROJECT_ID"
gcloud config set artifacts/location "$REGION"
gcloud config set run/region "$REGION"

echo "✅ GCP configuration set"

# === 3. CREATE GCS BUCKET ===
echo ""
echo "🪣 Creating GCS Bucket"
echo "--------------------"

echo "Creating bucket: gs://$BUCKET_NAME"
if gsutil ls -b "gs://$BUCKET_NAME" 2>/dev/null; then
    echo "✅ Bucket already exists: $BUCKET_NAME"
else
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$BUCKET_NAME"
    echo "✅ Bucket created successfully"
fi

# === 4. CREATE BIGQUERY DATASET ===
echo ""
echo "📊 Creating BigQuery Dataset"
echo "---------------------------"

echo "Creating dataset: $PROJECT_ID:$DATASET_ID"
if bq ls -d "$PROJECT_ID:$DATASET_ID" 2>/dev/null; then
    echo "✅ Dataset already exists: $DATASET_ID"
else
    bq --location="$REGION" mk --dataset "$PROJECT_ID:$DATASET_ID"
    echo "✅ Dataset created successfully"
fi

# === 5. DEPLOY CLOUD FUNCTION ===
echo ""
echo "⚡ Deploying Cloud Function"
echo "-------------------------"

FUNCTION_PATH="./Cloud_functions/gcs_to_bq"
if [ ! -d "$FUNCTION_PATH" ]; then
  echo "❌ Error: $FUNCTION_PATH directory not found!"
  echo "   Make sure you're running this script from the services/ directory"
  exit 1
fi

echo "Deploying function: $FUNCTION_NAME"
echo "  • Source: $FUNCTION_PATH"
echo "  • Trigger: gs://$BUCKET_NAME"
echo "  • Region: $REGION"

gcloud functions deploy "$FUNCTION_NAME" \
  --runtime python311 \
  --service-account="$SERVICE_ACCOUNT_EMAIL" \
  --trigger-resource="$BUCKET_NAME" \
  --trigger-event=google.storage.object.finalize \
  --entry-point=gcs_to_bq \
  --memory=1GiB \
  --timeout=540s \
  --region="$REGION" \
  --source="$FUNCTION_PATH" \
  --no-gen2 \
  --quiet

echo "✅ Cloud Function deployed successfully"

# === 6. CREATE ARTIFACT REGISTRY ===
echo ""
echo "📦 Setting up Artifact Registry"
echo "------------------------------"

echo "Creating repository: $ARTIFACT_REPO"
if gcloud artifacts repositories describe "$ARTIFACT_REPO" --location="$REGION" 2>/dev/null; then
    echo "✅ Artifact Registry repository already exists"
else
    gcloud artifacts repositories create "$ARTIFACT_REPO" \
      --repository-format=docker \
      --location="$REGION" \
      --description="Docker repository for ML services" \
      --quiet
    echo "✅ Artifact Registry repository created"
fi

# === 7. BUILD AND PUSH DOCKER IMAGES ===
echo ""
echo "🐳 Building and Pushing Docker Images"
echo "------------------------------------"

# Validate service directories
if [ ! -d "./ML" ]; then
    echo "❌ Error: ./ML directory not found!"
    echo "   Make sure ML service directory exists in services/"
    exit 1
fi

if [ ! -d "./Predict" ]; then
    echo "❌ Error: ./Predict directory not found!"
    echo "   Make sure Predict service directory exists in services/"
    exit 1
fi

TRAINING_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$TRAINING_SERVICE"
PREDICT_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$PREDICT_SERVICE"

echo "Building images:"
echo "  • Training Service: $TRAINING_IMAGE"
echo "  • Prediction Service: $PREDICT_IMAGE"

echo "🔨 Building Training Service..."
gcloud builds submit ./ML \
  --tag "$TRAINING_IMAGE" \
  --region="$REGION2" \
  --quiet &

echo "🔨 Building Prediction Service..."
gcloud builds submit ./Predict \
  --tag "$PREDICT_IMAGE" \
  --region="$REGION2" \
  --quiet &

echo "⏳ Waiting for builds to complete..."
wait

echo "✅ Docker images built and pushed successfully"

# === 8. DEPLOY CLOUD RUN SERVICES ===
echo ""
echo "🚀 Deploying Cloud Run Services"
echo "------------------------------"

echo "Deploying Training Service..."
gcloud run deploy "$TRAINING_SERVICE" \
  --image="$TRAINING_IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --memory=2Gi \
  --cpu=1 \
  --port=81 \
  --timeout=3600 \
  --concurrency=10 \
  --max-instances=5 \
  --allow-unauthenticated \
  --service-account="$SERVICE_ACCOUNT_EMAIL" \
  --quiet

echo "Deploying Prediction Service..."
gcloud run deploy "$PREDICT_SERVICE" \
  --image="$PREDICT_IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --memory=2Gi \
  --cpu=1 \
  --port=81 \
  --timeout=300 \
  --concurrency=100 \
  --max-instances=10 \
  --allow-unauthenticated \
  --service-account="$SERVICE_ACCOUNT_EMAIL" \
  --quiet

echo "✅ Cloud Run services deployed successfully"

# === 9. FETCH SERVICE URLS ===
echo ""
echo "🔗 Fetching Service URLs"
echo "-----------------------"

echo "Getting service URLs..."
TRAINING_URL=$(gcloud run services describe "$TRAINING_SERVICE" --region="$REGION" --format="value(status.url)")
PREDICT_URL=$(gcloud run services describe "$PREDICT_SERVICE" --region="$REGION" --format="value(status.url)")

if [ -z "$TRAINING_URL" ] || [ -z "$PREDICT_URL" ]; then
  echo "❌ Error: Failed to fetch service URLs"
  echo "   Training URL: $TRAINING_URL"
  echo "   Predict URL: $PREDICT_URL"
  exit 1
fi

echo "✅ Service URLs retrieved:"
echo "  • Training Service: $TRAINING_URL"
echo "  • Prediction Service: $PREDICT_URL"

# === 10. UPDATE CONFIGURATION FILES ===
echo ""
echo "📝 Updating Configuration Files"
echo "------------------------------"

# Update root .env file with service URLs
echo "Updating root .env with service URLs..."
sed -i.bak "s|TRAINING_URL=\"\"|TRAINING_URL=\"$TRAINING_URL\"|g" "$ENV_PATH"
sed -i.bak "s|PREDICT_URL=\"\"|PREDICT_URL=\"$PREDICT_URL\"|g" "$ENV_PATH"
rm "$ENV_PATH.bak" 2>/dev/null || true

echo "✅ Root .env file updated"

# Update prototype .env file with service URLs
PROTOTYPE_ENV_PATH="../prototype/.env"
echo "Updating prototype .env with service URLs..."

cat <<EOF > "$PROTOTYPE_ENV_PATH"
TRAINING_URL=$TRAINING_URL
PREDICT_URL=$PREDICT_URL
EOF

echo "✅ Prototype .env file updated"

# Update Streamlit secrets.toml
SECRETS_PATH="../prototype/.streamlit/secrets.toml"
if [ -f "$SECRETS_PATH" ]; then
    echo "Updating Streamlit secrets.toml..."
    
    # Use Python to update the TOML file safely
    python3 -c "
import re
import sys

secrets_path = '$SECRETS_PATH'
training_url = '$TRAINING_URL'
predict_url = '$PREDICT_URL'

try:
    with open(secrets_path, 'r') as f:
        content = f.read()
    
    # Update the cloud_run section URLs
    content = re.sub(r'ml_service_url = \"\"', f'ml_service_url = \"{training_url}\"', content)
    content = re.sub(r'predict_service_url = \"\"', f'predict_service_url = \"{predict_url}\"', content)
    
    with open(secrets_path, 'w') as f:
        f.write(content)
    
    print('✅ secrets.toml updated successfully')
except Exception as e:
    print(f'❌ Error updating secrets.toml: {e}')
    sys.exit(1)
"
else
    echo "⚠️  Warning: secrets.toml not found at $SECRETS_PATH"
    echo "   You may need to run setup.sh first"
fi

# === 11. DEPLOYMENT VERIFICATION ===
echo ""
echo "🔍 Verifying Deployment"
echo "----------------------"

echo "Testing service endpoints..."

# Test training service
echo "  🧪 Testing training service..."
if curl -s --max-time 10 "$TRAINING_URL/health" > /dev/null 2>&1; then
    echo "  ✅ Training service is responding"
else
    echo "  ⚠️  Training service health check failed (this is normal if health endpoint doesn't exist)"
fi

# Test prediction service
echo "  🧪 Testing prediction service..."
if curl -s --max-time 10 "$PREDICT_URL/health" > /dev/null 2>&1; then
    echo "  ✅ Prediction service is responding"
else
    echo "  ⚠️  Prediction service health check failed (this is normal if health endpoint doesn't exist)"
fi

# === 12. FINAL SUMMARY ===
echo ""
echo "🎉 Deployment Complete!"
echo "======================="
echo ""
echo "📊 Deployed Resources:"
echo "  • GCS Bucket: gs://$BUCKET_NAME"
echo "  • BigQuery Dataset: $PROJECT_ID:$DATASET_ID"
echo "  • Cloud Function: $FUNCTION_NAME (triggers on file upload)"
echo "  • Artifact Registry: $ARTIFACT_REPO"
echo "  • Training Service: $TRAINING_SERVICE"
echo "  • Prediction Service: $PREDICT_SERVICE"
echo ""
echo "🔗 Service URLs:"
echo "  • Training: $TRAINING_URL"
echo "  • Prediction: $PREDICT_URL"
echo ""
echo "📁 Updated Files:"
echo "  • Root .env (service URLs added)"
echo "  • prototype/.env (service URLs added)"
echo "  • prototype/.streamlit/secrets.toml (service URLs added)"
echo ""
echo "🚀 Next Steps:"
echo "  1. Test file upload to gs://$BUCKET_NAME"
echo "  2. Start your Streamlit app: cd ../prototype && streamlit run app.py"
echo "  3. Monitor logs: gcloud functions logs read $FUNCTION_NAME --region=$REGION"
echo ""
echo "✨ Your AutoML pipeline is ready to use!"