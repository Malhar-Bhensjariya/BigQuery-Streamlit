import streamlit as st
from google.cloud import bigquery, storage
from google.oauth2 import service_account

@st.cache_resource
def get_clients():
    """Initialize and cache GCP clients with proper error handling"""
    try:
        # Safely get credentials from secrets
        if "gcp_credentials" not in st.secrets:
            raise ValueError("Missing 'gcp_credentials' in secrets.toml")
            
        creds_dict = {
            "type": st.secrets["gcp_credentials"]["type"],
            "project_id": st.secrets["gcp_credentials"]["project_id"],
            "private_key": st.secrets["gcp_credentials"]["private_key"].replace('\\n', '\n'),
            "client_email": st.secrets["gcp_credentials"]["client_email"],
            "token_uri": "https://oauth2.googleapis.com/token"
        }

        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_info(creds_dict)

        # Optional values from secrets
        bucket_name = st.secrets.get("gcp_config", {}).get("bucket_name", "")
        gcloud_sdk_bin = st.secrets.get("gcp_config", {}).get("gcloud_sdk_bin", "")
        ml_service_url = st.secrets.get("cloud_run", {}).get("ml_service_url", "")
        predict_service_url = st.secrets.get("cloud_run", {}).get("predict_service_url", "")

        return {
            'bq': bigquery.Client(credentials=credentials),
            'gcs': storage.Client(credentials=credentials),
            'bucket': bucket_name,
            'gcloud_sdk_bin': gcloud_sdk_bin,
            'ml_url': ml_service_url,
            'predict_url': predict_service_url
        }

    except Exception as e:
        st.error(f"❌ Failed to initialize GCP clients: {str(e)}")
        st.stop()