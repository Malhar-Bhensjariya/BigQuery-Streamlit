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
            "token_uri": "https://oauth2.googleapis.com/token"  # Required field
        }

        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        
        return {
            'bq': bigquery.Client(credentials=credentials),
            'gcs': storage.Client(credentials=credentials),
            'bucket': st.secrets.get("gcp_config", {}).get("bucket_name", "")
        }
        
    except Exception as e:
        st.error(f"❌ Failed to initialize GCP clients: {str(e)}")
        st.stop()  # Halt the app if credentials are invalid