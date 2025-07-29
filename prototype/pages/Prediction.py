import streamlit as st
import subprocess
import os
from components.navigation import back_button
from components.train_model import show_training_popup
from components.prediction_form import show_prediction_popup

# Paths to Google Cloud SDK
GCLOUD_SDK_BIN = r"D:\AppData\Google\Cloud SDK\google-cloud-sdk\bin"
GSUTIL_PATH = os.path.join(GCLOUD_SDK_BIN, "gsutil.cmd")
BUCKET_NAME = "my-smart-ingest-bucket"

def verify_gsutil():
    """Check if gsutil is available"""
    if not os.path.exists(GSUTIL_PATH):
        st.error(f"❌ gsutil not found at {GSUTIL_PATH}")
        return False
    return True

def fetch_models(dataset, table):
    """Fetch available .pkl models from GCS"""
    try:
        models_path = f"gs://{BUCKET_NAME}/models/{dataset}/{table}/"
        
        # Run gsutil ls command to list .pkl files
        result = subprocess.run(
            [GSUTIL_PATH, 'ls', f"{models_path}*.pkl"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            # Extract model filenames from the full paths
            model_paths = result.stdout.strip().split('\n')
            models = []
            for path in model_paths:
                if path.strip():  # Skip empty lines
                    filename = path.split('/')[-1]
                    models.append({
                        'filename': filename,
                        'full_path': path.strip()
                    })
            return models
        else:
            return []
            
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

def show():
    st.title("Model Training & Prediction")
    back_button()
    
    # Initialize session state variables if they don't exist
    if 'show_train_form' not in st.session_state:
        st.session_state.show_train_form = False
    if 'is_retrain' not in st.session_state:
        st.session_state.is_retrain = False
    if 'show_prediction_form' not in st.session_state:
        st.session_state.show_prediction_form = False
    
    # Validate session state
    required_keys = ['selected_dataset', 'selected_table']
    if not all(key in st.session_state for key in required_keys):
        st.error("Missing required session state variables")
        return
    
    dataset = st.session_state.selected_dataset
    table = st.session_state.selected_table
    
    st.info(f"""
    **Dataset:** `{dataset}`  
    **Table:** `{table}`  
    **Models Path:** `models/{dataset}/{table}/`
    """)
    
    if not verify_gsutil():
        return
    
    # Show training popup if requested
    if st.session_state.show_train_form:
        show_training_popup()
        return  # Don't show the rest of the page while training form is active
    
    # Show prediction popup if requested
    if st.session_state.show_prediction_form:
        show_prediction_popup()
        return  # Don't show the rest of the page while prediction form is active
    
    # Show success message if training just completed
    if 'training_result' in st.session_state:
        st.success("🎉 Model training completed successfully! Refreshing available models...")
        # Clear the training result to avoid showing it again
        del st.session_state.training_result
    
    # Fetch and display models
    with st.spinner("Fetching available models..."):
        models = fetch_models(dataset, table)
    
    if models:
        st.subheader("📦 Available Models")
        
        # Display models in a nice format
        for i, model in enumerate(models):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{model['filename']}**")
                    st.caption(f"Path: `{model['full_path']}`")
                
                with col2:
                    if st.button("Select", key=f"select_model_{i}"):
                        st.session_state.selected_model = model
                        st.success(f"Selected: {model['filename']}")
                
                st.divider()
        
        # Action buttons
        st.subheader("Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Retrain Model", type="secondary", use_container_width=True):
                st.session_state.show_train_form = True
                st.session_state.is_retrain = True
                st.rerun()
        
        with col2:
            if st.button("Make Prediction", type="primary", use_container_width=True):
                if 'selected_model' in st.session_state:
                    st.session_state.show_prediction_form = True
                    st.rerun()
                else:
                    st.warning("Please select a model first.")
    
    else:
        st.warning("No models found in the specified path.")
        st.info("Models should be located at: `models/{dataset}/{table}/*.pkl`")
        
        # Option to train first model
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Train New Model", type="primary", use_container_width=True):
                st.session_state.show_train_form = True
                st.session_state.is_retrain = False
                st.rerun()