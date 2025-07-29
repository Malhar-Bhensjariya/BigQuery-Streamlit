#!/bin/bash

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="asia-south1"
REGION2="asia-east1"
FUNCTION_NAME="gcs-to-bq-pipeline"
DATASET_ID="dataset1"
BUCKET_NAME="my-smart-ingest-bucket"
TRAINING_SERVICE="ml-training-service" 
PREDICT_SERVICE="ml-predict-service"
ARTIFACT_REPO="ml-service-repo" 
SERVICE_ACCOUNT_EMAIL="gcs-bq-pipeline-sa@bigdata-sprint.iam.gserviceaccount.com"

echo "🚀 Initializing deployment..."

# Check if the project is set correctly
if [ -z "$PROJECT_ID" ]; then
  echo "❌ Error: GCP project is not set. Please run 'gcloud config set project PROJECT_ID' to set your project."
  exit 1
fi

# Create GCS bucket (if not exists)
echo "🪣 Creating GCS bucket..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME 2>/dev/null || echo "Bucket may already exist"

# Create BigQuery dataset
echo "📊 Creating BigQuery dataset..."
bq --location=$REGION mk --dataset $PROJECT_ID:$DATASET_ID 2>/dev/null || echo "Dataset may already exist"

# Check if Cloud Function directory exists
if [ ! -d "./Cloud_functions/gcs_to_bq" ]; then
    echo "❌ Error: ./Cloud_functions/gcs_to_bq directory not found!"
    echo "Please ensure the Cloud_functions/gcs_to_bq directory exists."
    exit 1
fi

# Deploy Cloud Function with environment variables directly (no .env files needed)
echo "⚡ Deploying Cloud Function..."
if gcloud functions deploy $FUNCTION_NAME \
  --runtime python311 \
  --service-account=$SERVICE_ACCOUNT_EMAIL \
  --trigger-resource=$BUCKET_NAME \
  --trigger-event=google.storage.object.finalize \
  --entry-point=gcs_to_bq \
  --memory=1GiB \
  --timeout=540s \
  --region=$REGION \
  --source=./Cloud_functions/gcs_to_bq \
  --no-gen2 \
  --quiet; then
    echo "✅ Cloud Function deployed successfully"
else
    echo "❌ Failed to deploy Cloud Function."
    exit 1
fi

# Create Artifact Registry repository (if not exists)
echo "📦 Creating Artifact Registry repo..."
gcloud artifacts repositories create $ARTIFACT_REPO \
  --repository-format=docker \
  --location=$REGION \
  --quiet 2>/dev/null || echo "Repository may already exist"

gcloud config set artifacts/location $REGION
gcloud config set run/region $REGION

# Check if ML directory exists
if [ ! -d "./ML" ]; then
    echo "❌ Error: ./ML directory not found!"
    echo "Please ensure the ML directory exists in the current path."
    exit 1
fi

# Build and push Docker image for Training Service
echo "🐳 Building and pushing Docker image for Training Service..."
gcloud builds submit ./ML \
  --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$TRAINING_SERVICE" --region=$REGION2 &

# Build and push Docker image for Prediction Service
echo "🐳 Building and pushing Docker image for Prediction Service..."
gcloud builds submit ./Predict \
  --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$PREDICT_SERVICE" --region=$REGION2 &

wait
echo "✅ Both Docker images have been built and pushed successfully"

# Deploy Training Service to Cloud Run
echo "🚀 Deploying Training Service to Cloud Run..."
if gcloud run deploy $TRAINING_SERVICE \
  --image="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$TRAINING_SERVICE" \
  --region=$REGION \
  --platform=managed \
  --memory=2Gi \
  --allow-unauthenticated \
  --port=81 \
  --quiet; then
    echo "✅ Cloud Run Training Service deployed successfully"
else
    echo "❌ Failed to deploy Cloud Run Training Service."
    exit 1
fi

# Deploy Prediction Service to Cloud Run
echo "🚀 Deploying Prediction Service to Cloud Run..."
if gcloud run deploy $PREDICT_SERVICE \
  --image="$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REPO/$PREDICT_SERVICE" \
  --region=$REGION \
  --platform=managed \
  --memory=2Gi \
  --allow-unauthenticated \
  --quiet; then
    echo "✅ Cloud Run Prediction Service deployed successfully"
else
    echo "❌ Failed to deploy Cloud Run Prediction Service."
    exit 1
fi

# Get Cloud Run Training Service URL
echo "🔗 Getting Cloud Run Service URL..."
TRAINING_URL=$(gcloud run services describe $TRAINING_SERVICE \
  --region=$REGION \
  --format="value(status.url)" 2>/dev/null)

PREDICT_URL=$(gcloud run services describe $PREDICT_SERVICE \
  --region=$REGION \
  --format="value(status.url)" 2>/dev/null)

if [ -z "$TRAINING_URL" ]; then
    echo "❌ Failed to get Cloud Run Training Service URL."
    exit 1
fi

echo "📋 Cloud Run Training Service URL: $TRAINING_URL"
echo "📋 Cloud Run Predicitng Service URL: $PREDICT_URL"

# Final Summary
echo "✅ Deployment complete!"
echo "🔗 Cloud Run Training Service URL: $TRAINING_URL"
echo "🔗 Cloud Run Predicting Service URL: $PREDICT_URL"
echo "📝 Environment variables have been set directly in the Cloud Function"
echo ""
echo "📋 Summary:"
echo "  - GCS Bucket: gs://$BUCKET_NAME"
echo "  - BigQuery Dataset: $PROJECT_ID:$DATASET_ID"
echo "  - Cloud Run Training Service: $TRAINING_SERVICE"
echo "  - Cloud Run Prediction Service: $PREDICT_SERVICE"
echo "  - Cloud Function: $FUNCTION_NAME"
