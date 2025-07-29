import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Prediction API endpoint
P_URL = os.environ.get('PREDICT_URL')

import requests
import streamlit as st

def show_prediction_form():
    """
    Display the prediction form based on session state
    """
    
    # Get required session state variables
    dataset = st.session_state.get('selected_dataset')
    table = st.session_state.get('selected_table')
    selected_model = st.session_state.get('selected_model')
    
    if not all([dataset, table, selected_model]):
        st.error("Missing required information. Please select a model first.")
        return
    
    # Extract model path from selected model
    model_path = selected_model['full_path'].replace(f"gs://my-smart-ingest-bucket/", "")
    
    title = "Make Prediction"
    st.subheader(title)
    
    # Fetch form metadata if not already loaded
    if 'form_metadata' not in st.session_state:
        with st.spinner("Loading form fields..."):
            try:
                response = requests.get(
                    f"{P_URL}/form_metadata",
                    params={
                        "bucket": "my-smart-ingest-bucket",
                        "model_path": model_path
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    st.session_state.form_metadata = response.json()
                else:
                    st.error(f"Failed to load form metadata: {response.text}")
                    return
                    
            except Exception as e:
                st.error(f"Error loading form metadata: {str(e)}")
                return
    
    metadata = st.session_state.form_metadata
    
    if metadata.get('status') != 'success':
        st.error("Failed to load model metadata")
        return
    
    with st.form(key="prediction_form", clear_on_submit=False):
        st.info(f"""
        **Model:** `{selected_model['filename']}`  
        **Problem Type:** `{metadata.get('problem_type', 'Unknown').replace('_', ' ').title()}`  
        **Target Column:** `{metadata.get('target_column', 'Unknown')}`
        """)
        
        # Display model information
        with st.expander("Model Information", expanded=False):
            model_info = metadata.get('model_info', {})
            
            col1, col2 = st.columns(2)
            with col1:
                if 'training_date' in model_info:
                    st.write(f"**Training Date:** {model_info['training_date'][:10]}")
                
                metrics = model_info.get('metrics', {})
                if 'accuracy' in metrics:
                    st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
                    
            with col2:
                if 'precision' in metrics:
                    st.write(f"**Precision:** {metrics['precision']:.3f}")
                if 'recall' in metrics:
                    st.write(f"**Recall:** {metrics['recall']:.3f}")
                if 'f1' in metrics:
                    st.write(f"**F1 Score:** {metrics['f1']:.3f}")
        
        # Generate form fields
        st.write("### Input Features")
        
        form_fields = metadata.get('form_fields', {})
        prediction_data = {}
        
        # Create form fields dynamically
        for field_name, field_config in form_fields.items():
            field_type = field_config.get('type')
            input_type = field_config.get('input_type')
            
            # Display field label and description
            display_name = field_name.replace('_', ' ').title()
            st.write(f"**{display_name}**")
            
            if field_config.get('description'):
                st.caption(field_config['description'])
            
            if input_type == 'number':
                min_val = field_config.get('min')
                max_val = field_config.get('max')
                default_val = field_config.get('default_value', min_val if min_val is not None else 0)
                
                # Determine decimal places based on the range and values
                if min_val is not None and max_val is not None:
                    range_val = max_val - min_val
                    if range_val < 1:
                        step = 0.01
                        format_str = "%.2f"
                    elif range_val < 10:
                        step = 0.1
                        format_str = "%.1f"
                    else:
                        step = 1.0
                        format_str = "%.0f"
                else:
                    step = 0.1
                    format_str = "%.1f"
                
                # Initialize session state for field if not exists
                field_key = f"pred_{field_name}"
                if field_key not in st.session_state:
                    st.session_state[field_key] = float(default_val)
                
                # Only use slider if min_val and max_val are defined
                if min_val is not None and max_val is not None:
                    slider_value = st.slider(
                        label=f"Select {display_name}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=st.session_state[field_key],
                        step=step,
                        key=f"slider_{field_name}",
                        format=format_str
                    )
                    prediction_data[field_name] = slider_value
                    st.session_state[field_key] = slider_value
                else:
                    # Fallback: handle case without valid slider range
                    st.warning(f"Skipping {display_name} as no range is defined for slider.")
                    prediction_data[field_name] = None  # Or skip the field from the prediction data
            
            elif input_type == 'select':
                options = field_config.get('options', [])
                default_val = field_config.get('default_value')
                
                # Find index of default value
                try:
                    default_index = options.index(default_val) if default_val in options else 0
                except (ValueError, IndexError):
                    default_index = 0
                
                value = st.selectbox(
                    label=f"Select {display_name}",
                    options=options,
                    index=default_index,
                    key=f"pred_{field_name}"
                )
                prediction_data[field_name] = value
        
        # Configuration options
        st.write("### Prediction Options")
        show_probs = st.checkbox(
            "Show Probabilities", 
            value=metadata.get('problem_type') != 'regression',
            disabled=metadata.get('problem_type') == 'regression',
            help="Display prediction probabilities (only for classification problems)"
        )
        
        # Form buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("🚀 Make Prediction", type="primary", use_container_width=True)
        
        with col2:
            reset_form = st.form_submit_button("🔄 Reset Form", use_container_width=True)
        
        with col3:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
    
    # Handle form submission - OUTSIDE the form
    if submitted:
        make_prediction(model_path, prediction_data, show_probs, metadata)
    
    if reset_form:
        # Clear form-related session state
        keys_to_clear = [key for key in st.session_state.keys() if key.startswith('pred_') or key.startswith('slider_') or key.startswith('number_')]
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Clear form metadata to reload defaults
        if 'form_metadata' in st.session_state:
            del st.session_state.form_metadata
        st.rerun()
    
    if cancel:
        st.session_state.show_prediction_form = False
        st.rerun()



def make_prediction(model_path, prediction_data, show_probs, metadata):
    """
    Send prediction request to the API
    
    Args:
        model_path (str): Path to the model
        prediction_data (dict): Input data for prediction
        show_probs (bool): Whether to show probabilities
        metadata (dict): Model metadata
    """
    
    with st.container():
        st.write("### Prediction in Progress...")
        
        # Create progress indicators
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_placeholder.info("Preparing prediction request...")
            progress_bar.progress(25)
            
            # Prepare prediction request
            prediction_request = {
                "bucket": "my-smart-ingest-bucket",
                "model_path": model_path,
                "data": prediction_data,
                "return_probs": show_probs and metadata.get('problem_type') != 'regression'
            }
            
            status_placeholder.info("Sending request to prediction service...")
            progress_bar.progress(50)
            
            # Make API call
            response = requests.post(
                f"{P_URL}/predict",
                json=prediction_request,
                timeout=30
            )
            
            progress_bar.progress(75)
            status_placeholder.info("Processing prediction...")
            
            if response.status_code == 200:
                result = response.json()
                progress_bar.progress(100)
                
                if result.get('status') == 'success':
                    # Show success message
                    st.success("✅ Prediction completed successfully!")
                    
                    prediction = result['prediction']
                    
                    # Display prediction result
                    st.write("### Prediction Result")
                    
                    problem_type = metadata.get('problem_type', '')
                    target_column = metadata.get('target_column', 'target')
                    target_display = target_column.replace('_', ' ').title()
                    
                    # Format prediction display based on problem type
                    if problem_type == 'regression':
                        # For regression, show the predicted value
                        predicted_value = prediction if isinstance(prediction, (int, float)) else float(prediction)
                        
                        # Create a generic interpretation
                        st.success(f"**Predicted {target_display}: {predicted_value:.2f}**")
                        st.info(f"The model predicts **{predicted_value:.2f}** for {target_display}")
                    
                    else:
                        # Classification
                        prediction_label = str(prediction[0] if isinstance(prediction, list) else prediction)
                        
                        # Get probabilities for better interpretation
                        probs = result.get('probabilities', [])
                        
                        if problem_type == 'binary_classification':
                            prob_positive = probs if isinstance(probs, (int, float)) else probs[1] if isinstance(probs, list) else 0.5
                            prob_negative = 1 - prob_positive
                            confidence = max(prob_positive, prob_negative)
                            
                            # Determine prediction based on probability, not raw prediction value
                            is_positive_prediction = prob_positive > 0.6
                            
                            # Generic binary classification interpretation
                            if is_positive_prediction:
                                st.success(f"✅ **{target_display}: POSITIVE/YES** (Confidence: {confidence:.1%})")
                                st.info(f"The model predicts a positive outcome for {target_display}")
                            else:
                                st.error(f"❌ **{target_display}: NEGATIVE/NO** (Confidence: {confidence:.1%})")
                                st.info(f"The model predicts a negative outcome for {target_display}")
                        
                        else:
                            # Multi-class classification
                            if isinstance(probs, list) and len(probs) > 0:
                                max_prob = max(probs)
                                confidence = max_prob
                            else:
                                confidence = 0.0
                            
                            # Generic multi-class interpretation
                            st.success(f"🎯 **{target_display}: {prediction_label.upper()}** (Confidence: {confidence:.1%})")
                            st.info(f"The model predicts **{prediction_label}** as the most likely class for {target_display}")
                    
                    # Show detailed technical results in expandable section
                    with st.expander("Detailed Technical Results", expanded=False):
                        st.write("**Raw Prediction Output:**")
                        if problem_type == 'regression':
                            st.code(f"Predicted Value: {prediction}")
                        else:
                            st.code(f"Predicted Class: {prediction}")
                        
                        # Show probabilities if available
                        if 'probabilities' in result and show_probs:
                            st.write("**Prediction Probabilities:**")
                            probs = result['probabilities']
                            
                            if problem_type == 'binary_classification':
                                prob_positive = probs if isinstance(probs, (int, float)) else probs[1] if isinstance(probs, list) else 0.5
                                prob_negative = 1 - prob_positive
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Probability (Class 0/Negative)", f"{prob_negative:.3%}")
                                with col2:
                                    st.metric("Probability (Class 1/Positive)", f"{prob_positive:.3%}")
                                    
                                # Progress bar for confidence
                                st.progress(prob_positive, text=f"Raw Probability: {prob_positive:.3%}")
                                
                            else:
                                # Multi-class classification
                                if isinstance(probs, list):
                                    for i, prob in enumerate(probs):
                                        st.metric(f"Class {i} Probability", f"{prob:.3%}")
                    
                    # Show input summary
                    with st.expander("Input Summary", expanded=False):
                        for key, value in prediction_data.items():
                            display_key = key.replace('_', ' ').title()
                            st.write(f"**{display_key}:** {value}")
                
                else:
                    st.error(f"Prediction failed: {result.get('message', 'Unknown error')}")
                    progress_bar.empty()
            
            else:
                # Handle HTTP errors
                try:
                    error_response = response.json()
                    error_detail = error_response.get('detail', f'HTTP {response.status_code}')
                except:
                    error_detail = f"HTTP {response.status_code}: {response.text}"
                
                st.error(f"Prediction request failed: {error_detail}")
                progress_bar.empty()
        
        except requests.exceptions.Timeout:
            st.error("⏰ Prediction request timed out. Please try again.")
            progress_bar.empty()
        
        except requests.exceptions.ConnectionError:
            st.error("🌐 Could not connect to prediction service. Please check if the service is running.")
            progress_bar.empty()
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            progress_bar.empty()


def show_prediction_popup():
    """
    Show the prediction form popup based on session state
    """
    if st.session_state.get('show_prediction_form', False):
        show_prediction_form()
    else:
        st.info("No prediction form to display")