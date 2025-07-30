import streamlit as st
import os
import tempfile
import subprocess
from components.navigation import back_button
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Paths to Google Cloud SDK (update these if needed)
GCLOUD_SDK_BIN = os.environ.get("GCLOUD_SDK_BIN")
if not GCLOUD_SDK_BIN:
    st.warning("⚠️ GCLOUD_SDK_BIN is not set. Please check your .env file.")
GSUTIL_PATH = os.path.join(GCLOUD_SDK_BIN, "gsutil.cmd")
BUCKET_NAME = "my-smart-ingest-bucket"  # From secrets.toml

def verify_gsutil():
    """Check if gsutil is available"""
    if not os.path.exists(GSUTIL_PATH):
        st.error(f"❌ gsutil not found at {GSUTIL_PATH}")
        return False
    return True

def upload_to_gcs(file, dataset, table, mode):
    """Upload file using gsutil with strict naming"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        # Generate filename
        filename = f"{dataset}-{table}-{mode}__{file.name}"
        
        # Run gsutil command
        result = subprocess.run(
            [GSUTIL_PATH, 'cp', tmp_path, f'gs://{BUCKET_NAME}/{filename}'],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return result, filename
        
    except Exception as e:
        return None, str(e)

def show():
    st.title(f"{st.session_state.mode.capitalize()} to Table")
    back_button()
    
    # Validate session state
    required_keys = ['selected_dataset', 'selected_table', 'mode']
    if not all(key in st.session_state for key in required_keys):
        st.error("Missing required session state variables")
        return
        
    if st.session_state.mode not in ['create', 'append']:
        st.error("Invalid mode selected")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Select CSV file", 
        type=["csv"],
        help="File will be uploaded as: dataset-table-mode__filename.csv"
    )
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        
        # Show file info
        st.info(f"""
        **Dataset:** `{st.session_state.selected_dataset}`  
        **Table:** `{st.session_state.selected_table}`  
        **Mode:** `{st.session_state.mode}`  
        **Filename:** `{uploaded_file.name}`
        """)
        
        # Show generated filename
        generated_name = (
            f"{st.session_state.selected_dataset}-"
            f"{st.session_state.selected_table}-"
            f"{st.session_state.mode}__"
            f"{uploaded_file.name}"
        )
        st.code(f"Will be uploaded as: {generated_name}", language="text")
        
        # Upload button
        if st.button("🚀 Upload File", type="primary"):
            if not verify_gsutil():
                return
                
            with st.spinner("Uploading file..."):
                result, filename = upload_to_gcs(
                    uploaded_file,
                    st.session_state.selected_dataset,
                    st.session_state.selected_table,
                    st.session_state.mode
                )
                
            if result and result.returncode == 0:
                st.success(f"✅ File uploaded successfully as: `{filename}`")
                st.code(result.stdout, language="text")
            else:
                st.error("❌ Upload failed")
                if result:
                    st.error(result.stderr)
                else:
                    st.error(filename)