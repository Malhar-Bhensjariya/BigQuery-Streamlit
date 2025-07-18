import toml
from google.cloud import bigquery

# Load secrets.toml
with open('.streamlit/secrets.toml') as f:
    secrets = toml.load(f)

# Get credentials
creds = secrets['gcp_credentials']
creds['private_key'] = creds['private_key'].replace('\\n', '\n')  # Fix newlines

# Test connection
client = bigquery.Client.from_service_account_info(creds)
print("Connected! Available datasets:")
print(list(client.list_datasets()))