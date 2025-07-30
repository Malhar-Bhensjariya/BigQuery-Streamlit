#!/bin/bash
set -e

echo "🚀 BigQuery Streamlit AutoML Setup"
echo "=================================="

# === 1. PROJECT CONFIGURATION ===
echo ""
echo "📋 Project Configuration"
echo "------------------------"

# Get current project or ask for one
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
if [ -n "$CURRENT_PROJECT" ]; then
    read -p "Use current GCP project '$CURRENT_PROJECT'? (y/n): " use_current
    if [[ $use_current =~ ^[Yy]$ ]]; then
        PROJECT_ID="$CURRENT_PROJECT"
    else
        read -p "Enter your GCP Project ID: " PROJECT_ID
        gcloud config set project "$PROJECT_ID"
    fi
else
    read -p "Enter your GCP Project ID: " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

echo "✅ Using project: $PROJECT_ID"

# === 2. REGION CONFIGURATION ===
echo ""
echo "🌍 Region Configuration"
echo "----------------------"
echo "You need to specify two regions:"
echo "  - Primary region (for most services)"
echo "  - Secondary region (for Cloud Build)"
echo ""

read -p "Enter primary region (default: asia-south1): " PRIMARY_REGION
PRIMARY_REGION=${PRIMARY_REGION:-"asia-south1"}

read -p "Enter secondary region (default: asia-east1): " SECONDARY_REGION
SECONDARY_REGION=${SECONDARY_REGION:-"asia-east1"}

echo "✅ Primary region: $PRIMARY_REGION"
echo "✅ Secondary region: $SECONDARY_REGION"

# === 3. SERVICE ACCOUNT SETUP ===
echo ""
echo "👤 Service Account Setup"
echo "-----------------------"

SERVICE_ACCOUNT_NAME="bq-streamlit-app"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# Check if service account exists
if gcloud iam service-accounts list --filter="email:$SERVICE_ACCOUNT_EMAIL" --format="value(email)" | grep -q "$SERVICE_ACCOUNT_EMAIL"; then
    echo "✅ Service account already exists: $SERVICE_ACCOUNT_NAME"
else
    echo "🔧 Creating service account: $SERVICE_ACCOUNT_NAME"
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="BigQuery Streamlit AutoML Service Account" \
        --description="Service account for BigQuery Streamlit AutoML application"
    echo "✅ Service account created successfully"
fi

# === 4. IAM PERMISSIONS ===
echo ""
echo "🔐 Assigning IAM Permissions"
echo "---------------------------"

ROLES=(
    "roles/aiplatform.admin"
    "roles/aiplatform.user"
    "roles/bigquery.dataEditor"
    "roles/bigquery.user"
    "roles/bigquery.admin"
    "roles/storage.objectAdmin"
    "roles/storage.objectViewer"
    "roles/cloudfunctions.developer"
    "roles/run.admin"
    "roles/artifactregistry.admin"
    "roles/cloudbuild.builds.editor"
)

echo "Assigning roles to service account..."
for ROLE in "${ROLES[@]}"; do
    echo "  🔗 Binding role: $ROLE"
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="$ROLE" \
        --quiet
done
echo "✅ All IAM roles assigned successfully"

# === 5. SERVICE ACCOUNT KEY GENERATION ===
echo ""
echo "🔑 Generating Service Account Key"
echo "--------------------------------"

KEY_FILE="service-account-key.json"
if [ -f "$KEY_FILE" ]; then
    read -p "Service account key file exists. Regenerate? (y/n): " regenerate_key
    if [[ $regenerate_key =~ ^[Yy]$ ]]; then
        rm "$KEY_FILE"
        gcloud iam service-accounts keys create "$KEY_FILE" \
            --iam-account="$SERVICE_ACCOUNT_EMAIL"
        echo "✅ New service account key generated"
    else
        echo "✅ Using existing service account key"
    fi
else
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SERVICE_ACCOUNT_EMAIL"
    echo "✅ Service account key generated: $KEY_FILE"
fi

# === 6. PYTHON DEPENDENCIES ===
echo ""
echo "🐍 Installing Python Dependencies"
echo "--------------------------------"

if [ -f "prototype/requirements.txt" ]; then
    echo "Installing dependencies from prototype/requirements.txt..."
    pip install -r prototype/requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  Warning: prototype/requirements.txt not found"
    echo "   Make sure to install dependencies manually later"
fi


ARTIFACT_REPO=ml-service-repo

# Service URLs (will be populated after deployment)
TRAINING_URL=""
PREDICT_URL=""
EOF
echo "✅ .env file created"

# === 8. STREAMLIT SECRETS CONFIGURATION ===
echo ""
echo "Creating Streamlit secrets.toml..."

# Create .streamlit directory
mkdir -p prototype/.streamlit

# Extract service account details from JSON key
if [ -f "$KEY_FILE" ]; then
    # Parse JSON using python (more reliable than jq which might not be installed)
    python3 -c "
import json
import sys

try:
    with open('$KEY_FILE', 'r') as f:
        data = json.load(f)
    
    # Create secrets.toml content
    secrets_content = f'''[gcp_credentials]
type = \"{data['type']}\"
project_id = \"{data['project_id']}\"
private_key_id = \"{data['private_key_id']}\"
private_key = \"\"\"{data['private_key']}\"\"\"
client_email = \"{data['client_email']}\"
client_id = \"{data['client_id']}\"
auth_uri = \"{data['auth_uri']}\"
token_uri = \"{data['token_uri']}\"
auth_provider_x509_cert_url = \"{data['auth_provider_x509_cert_url']}\"
client_x509_cert_url = \"{data['client_x509_cert_url']}\"

[gcp_config]
project_id = \"$PROJECT_ID\"
region = \"$PRIMARY_REGION\"
dataset_id = \"dataset1\"
bucket_name = \"my-smart-ingest-bucket\"

[cloud_run]
ml_service_url = \"\"
predict_service_url = \"\"
'''

    with open('prototype/.streamlit/secrets.toml', 'w') as f:
        f.write(secrets_content)
    
    print('✅ secrets.toml created successfully')
except Exception as e:
    print(f'❌ Error creating secrets.toml: {e}')
    sys.exit(1)
"
else
    echo "❌ Error: Service account key file not found"
    exit 1
fi

# === 9. ENABLE REQUIRED APIS ===
echo ""
echo "🔧 Enabling Required Google Cloud APIs"
echo "-------------------------------------"

APIS=(
    "bigquery.googleapis.com"
    "storage.googleapis.com"
    "cloudfunctions.googleapis.com"
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "aiplatform.googleapis.com"
)

echo "Enabling required APIs..."
for API in "${APIS[@]}"; do
    echo "  🔌 Enabling: $API"
    gcloud services enable "$API" --quiet
done
echo "✅ All required APIs enabled"

# === 10. FINAL SUMMARY ===
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Configuration Summary:"
echo "  • Project ID: $PROJECT_ID"
echo "  • Primary Region: $PRIMARY_REGION"
echo "  • Secondary Region: $SECONDARY_REGION"
echo "  • Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "  • Service Account Key: $KEY_FILE"
echo ""
echo "📁 Files Created:"
echo "  • .env (environment variables)"
echo "  • prototype/.streamlit/secrets.toml (Streamlit configuration)"
echo "  • $KEY_FILE (service account credentials)"
echo ""
echo "⚠️  Security Note:"
echo "  • Keep $KEY_FILE secure and never commit it to version control"
echo "  • Add $KEY_FILE to your .gitignore file"
echo ""
echo "🚀 Next Steps:"
echo "  1. Review the configuration files"
echo "  2. Run: cd services && bash deploy.sh"
echo "  3. Start your Streamlit app: cd prototype && streamlit run app.py"