import streamlit as st
import requests
import json
from time import sleep
from gcp_clients import get_clients
from dotenv import load_dotenv
import os
load_dotenv()

clients = get_clients()

T_URL = clients.get("ml_url") or os.getenv("TRAINING_URL")

if not T_URL:
    st.warning("⚠️ TRAINING_URL not found in secrets.toml or .env")


# Default configuration
DEFAULT_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42,
    'epochs': 30,
    'batch_size': 32,
    'early_stopping_patience': 10,
    'model_params': {
        'hidden_layers': [64, 32],
        'dropout_rate': 0.2,
        'l2_reg': 0.01,
        'learning_rate': 0.001
    }
}

def show_train_model_form(project_id, bucket_name, dataset_id, table_id, is_retrain=False):
    """
    Display the training form as a modal/popup
    
    Args:
        project_id (str): GCP project ID
        bucket_name (str): GCS bucket name
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table ID
        is_retrain (bool): Whether this is retraining or first training
    """
    
    title = "🔄 Retrain Model" if is_retrain else "🚀 Train First Model"
    st.subheader(title)
    
    with st.form(key="train_model_form", clear_on_submit=False):
        st.info(f"""
        **Project:** `{project_id}`  
        **Bucket:** `{bucket_name}`  
        **Dataset:** `{dataset_id}`  
        **Table:** `{table_id}`
        """)
        
        # Target column input (optional)
        st.markdown("### Target Column")
        target_column = st.text_input(
            "Target Column (optional)",
            placeholder="Leave empty for auto-detection",
            help="Specify the target column name. If left empty, the system will auto-detect the target column."
        )
        
        # Configuration section
        st.markdown("### Training Configuration (Optional)")
        
        # Expandable advanced configuration
        with st.expander("🔧 Advanced Configuration", expanded=False):
            st.markdown("**Training Parameters**")
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Size", 0.1, 0.4, DEFAULT_CONFIG['test_size'], 0.05)
                val_size = st.slider("Validation Size", 0.1, 0.4, DEFAULT_CONFIG['val_size'], 0.05)
                random_state = st.number_input("Random State", value=DEFAULT_CONFIG['random_state'], min_value=1)
                epochs = st.number_input("Epochs", value=DEFAULT_CONFIG['epochs'], min_value=1, max_value=200)
            
            with col2:
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
                early_stopping_patience = st.number_input("Early Stopping Patience", 
                                                        value=DEFAULT_CONFIG['early_stopping_patience'], 
                                                        min_value=1, max_value=50)
            
            st.markdown("**Model Parameters**")
            col3, col4 = st.columns(2)
            
            with col3:
                # Hidden layers configuration
                st.markdown("**Hidden Layers**")
                layer1 = st.number_input("Layer 1 Size", value=64, min_value=8, max_value=512, step=8)
                layer2 = st.number_input("Layer 2 Size", value=32, min_value=8, max_value=512, step=8)
                
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, DEFAULT_CONFIG['model_params']['dropout_rate'], 0.05)
            
            with col4:
                l2_reg = st.number_input("L2 Regularization", 
                                       value=DEFAULT_CONFIG['model_params']['l2_reg'], 
                                       min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
                learning_rate = st.number_input("Learning Rate", 
                                              value=DEFAULT_CONFIG['model_params']['learning_rate'], 
                                              min_value=0.0001, max_value=0.01, step=0.0001, format="%.4f")
            
            # Build custom config
            custom_config = {
                'test_size': test_size,
                'val_size': val_size,
                'random_state': int(random_state),
                'epochs': int(epochs),
                'batch_size': int(batch_size),
                'early_stopping_patience': int(early_stopping_patience),
                'model_params': {
                    'hidden_layers': [int(layer1), int(layer2)],
                    'dropout_rate': dropout_rate,
                    'l2_reg': l2_reg,
                    'learning_rate': learning_rate
                }
            }
            
            use_custom_config = st.checkbox("Use Custom Configuration", value=False)
        
        # Form buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("🚀 Start Training", type="primary", use_container_width=True)
        
        with col2:
            reset_form = st.form_submit_button("🔄 Reset Form", use_container_width=True)
        
        with col3:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
    
    # Handle form submission
    if submitted:
        train_model(project_id, bucket_name, dataset_id, table_id, 
                   target_column if target_column.strip() else None,
                   custom_config if use_custom_config else None)
    
    if reset_form:
        st.rerun()
    
    if cancel:
        st.session_state.show_train_form = False
        st.rerun()

def train_model(project_id, bucket_name, dataset_id, table_id, target_column=None, config=None):
    """
    Send training request to the API
    
    Args:
        project_id (str): GCP project ID
        bucket_name (str): GCS bucket name
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table ID
        target_column (str, optional): Target column name
        config (dict, optional): Training configuration
    """
    
    # Prepare request body
    request_body = {
        "project_id": project_id,
        "bucket_name": bucket_name,
        "dataset_id": dataset_id,
        "table_id": table_id
    }
    
    # Add optional parameters only if they are provided
    if target_column:
        request_body["target_column"] = target_column
    
    if config:
        request_body["config"] = config
    
    # Show training progress
    with st.container():
        st.markdown("### Training in Progress...")
        
        # Create placeholder for status updates
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        status_placeholder.info("Initializing training...")
        progress_bar.progress(10)
        sleep(2)
        
        status_placeholder.info("Loading data...")
        progress_bar.progress(20)
        sleep(5)
        
        status_placeholder.info("Sending training request...")
        progress_bar.progress(40)
        sleep(3)
        
        try:
            # Send POST request
            status_placeholder.info("Training the model...")
            progress_bar.progress(50)
            
            response = requests.post(
                f"{T_URL}/train",
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minutes timeout
            )
            
            progress_bar.progress(70)
            status_placeholder.info("⏳ Processing training request...")
            sleep(5)
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                progress_bar.progress(100)
                
                if result.get("status") == "success":
                    # Show success message
                    st.success("🎉 Training Completed Successfully!")
                    
                    # Display all metrics
                    st.subheader("📊 Model Performance Metrics")
                    metrics = result.get('metrics', {})
                    
                    # Create columns for metrics
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        if 'accuracy' in metrics:
                            st.metric(
                                label="Accuracy", 
                                value=f"{metrics['accuracy']:.4f}",
                                delta=f"{metrics['accuracy']*100:.2f}%"
                            )
                    
                    with metric_cols[1]:
                        if 'precision' in metrics:
                            st.metric(
                                label="Precision", 
                                value=f"{metrics['precision']:.4f}",
                                delta=f"{metrics['precision']*100:.2f}%"
                            )
                    
                    with metric_cols[2]:
                        if 'recall' in metrics:
                            st.metric(
                                label="Recall", 
                                value=f"{metrics['recall']:.4f}",
                                delta=f"{metrics['recall']*100:.2f}%"
                            )
                    
                    with metric_cols[3]:
                        if 'f1' in metrics:
                            st.metric(
                                label="F1 Score", 
                                value=f"{metrics['f1']:.4f}",
                                delta=f"{metrics['f1']*100:.2f}%"
                            )
                    
                    # Model details
                    st.subheader("🔧 Model Details")
                    detail_cols = st.columns(2)
                    
                    with detail_cols[0]:
                        st.metric(
                            label="Problem Type", 
                            value=result['problem_type'].replace('_', ' ').title()
                        )
                        st.metric(
                            label="Target Column", 
                            value=result['target_column']
                        )
                    
                    with detail_cols[1]:
                        st.metric(
                            label="Input Features", 
                            value=str(result['input_size'])
                        )
                        if result.get('num_classes'):
                            st.metric(
                                label="Classes", 
                                value=str(result['num_classes'])
                            )
                    
                    # Model path information
                    st.info(f"**Model saved at:** `{result['model_path']}`")
                    
                    # Done button
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("Done", type="primary", use_container_width=True):
                            # Store results in session state
                            st.session_state.training_result = result
                            st.session_state.show_train_form = False
                            st.rerun()
                else:
                    st.error(f"❌ Training failed: {result.get('message', 'Unknown error')}")
            
            else:
                # Handle HTTP errors
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get('detail', f'HTTP {response.status_code}')
                except:
                    error_detail = f"HTTP {response.status_code}: {response.text}"
                
                st.error(f"❌ Training failed: {error_detail}")
                progress_bar.empty()
        
        except requests.exceptions.Timeout:
            st.error("❌ Training request timed out. The training might still be running in the background.")
            progress_bar.empty()
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Failed to connect to the training service. Please check your internet connection.")
            progress_bar.empty()
        
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            progress_bar.empty()

def show_training_popup():
    """
    Show the training form popup based on session state
    """
    if st.session_state.get('show_train_form', False):
        # Get required parameters from session state
        project_id = "bigdata-sprint"  # You might want to get this from session state
        bucket_name = "my-smart-ingest-bucket"  # You might want to get this from session state
        dataset_id = st.session_state.get('selected_dataset', '')
        table_id = st.session_state.get('selected_table', '')
        is_retrain = st.session_state.get('is_retrain', False)
        
        if dataset_id and table_id:
            show_train_model_form(project_id, bucket_name, dataset_id, table_id, is_retrain)
        else:
            st.error("Missing dataset or table information")
            st.session_state.show_train_form = False