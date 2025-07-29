import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import torch
import logging
from google.cloud import storage
from typing import Tuple, Dict, Any, List, Union
from sklearn.compose import ColumnTransformer

# Import the model class from shared module
from model_definition import PyTorchModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
model_cache = {}

def download_blob(bucket_name: str, source_blob_name: str) -> str:
    """Download a blob from GCS to a temporary file"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        _, temp_path = tempfile.mkstemp()
        blob.download_to_filename(temp_path)
        logger.info(f"Downloaded model from gs://{bucket_name}/{source_blob_name}")
        return temp_path
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        raise RuntimeError(f"Model download failed: {str(e)}")

def get_original_features(feature_metadata: Dict) -> List[str]:
    """Extract original feature names from metadata, excluding encoded features"""
    original_features = []
    processed_features = set()
    
    for feature_name, meta in feature_metadata.items():
        # Skip encoded features
        if meta.get('is_encoded', False):
            continue
            
        orig_feature = meta.get("original_feature", feature_name)
        
        # Skip if already processed
        if orig_feature in processed_features:
            continue
            
        processed_features.add(orig_feature)
        original_features.append(orig_feature)
    
    return original_features

def load_model(bucket_name: str, model_path: str) -> Tuple[Any, Any, Dict, List]:
    """Robust model loading with metadata validation"""
    cache_key = f"{bucket_name}/{model_path}"
    
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    try:
        # Download model
        local_path = download_blob(bucket_name, model_path)
        with open(local_path, "rb") as f:
            model_package = pickle.load(f)
        
        # Load model
        model_params = model_package["model_params"]
        model = PyTorchModel(**model_params)
        model.load_state_dict(model_package["model_state"])
        model.eval()
        
        # Get components
        preprocessor = model_package["preprocessor"]
        metadata = model_package.get("metadata", {})
        
        # Validate and enhance feature metadata for backward compatibility
        feature_metadata = metadata.get("feature_metadata", {})
        feature_names = metadata.get("feature_names", [])
        
        # Ensure all processed features have proper metadata
        for feat in feature_names:
            if feat not in feature_metadata:
                # Try to infer from feature name patterns
                if feat.startswith('num__'):
                    # Numeric feature from ColumnTransformer
                    original_feature = feat.replace('num__', '')
                    feature_metadata[feat] = {
                        "original_feature": original_feature,
                        "type": "numeric",
                        "transformer": "numeric",
                        "is_encoded": False
                    }
                elif feat.startswith('cat__'):
                    # Categorical encoded feature from ColumnTransformer
                    parts = feat.replace('cat__', '').split('_', 1)
                    if len(parts) == 2:
                        original_feature, category_value = parts
                        feature_metadata[feat] = {
                            "original_feature": original_feature,
                            "type": "categorical_encoded",
                            "transformer": "categorical",
                            "is_encoded": True,
                            "category_value": category_value
                        }
                    else:
                        feature_metadata[feat] = {
                            "original_feature": feat,
                            "type": "unknown"
                        }
                else:
                    # Default metadata for unknown features
                    feature_metadata[feat] = {
                        "original_feature": feat,
                        "type": "unknown"
                    }
        
        metadata['feature_metadata'] = feature_metadata
        
        # Cache and return
        model_cache[cache_key] = (model, preprocessor, metadata, feature_names)
        os.unlink(local_path)
        
        return model, preprocessor, metadata, feature_names
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {str(e)}")

def validate_input(input_df: pd.DataFrame, expected_columns: List[str], metadata: dict) -> None:
    """Enhanced input validation against expected schema"""
    try:
        # Check for missing columns
        missing_cols = set(expected_columns) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {list(missing_cols)}")
        
        # Check for unexpected columns (warn but don't fail)
        unexpected_cols = set(input_df.columns) - set(expected_columns)
        if unexpected_cols:
            logger.warning(f"Unexpected columns will be ignored: {list(unexpected_cols)}")
        
        # Validate categorical values if available
        categorical_values = metadata.get('categorical_values', {})
        for col in input_df.columns:
            if col in categorical_values:
                invalid_values = set(input_df[col].dropna().astype(str)) - set(map(str, categorical_values[col]))
                if invalid_values:
                    raise ValueError(f"Invalid values for {col}: {list(invalid_values)}. "
                                   f"Expected one of: {categorical_values[col]}")
        
        logger.info("Input validation passed")
        
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise ValueError(f"Input validation failed: {str(e)}")

def predict_input(model: torch.nn.Module, 
                 preprocessor: ColumnTransformer, 
                 input_df: pd.DataFrame,
                 metadata: dict) -> Union[list, float, int]:
    """
    Make predictions on input data with enhanced output processing
    """
    try:
        # Get original features that the model expects
        feature_metadata = metadata.get('feature_metadata', {})
        original_features = get_original_features(feature_metadata)
        
        # Filter input to expected columns only and maintain order
        if original_features:
            # Ensure we only use columns that exist in the input
            available_features = [col for col in original_features if col in input_df.columns]
            input_filtered = input_df[available_features].copy()
        else:
            input_filtered = input_df.copy()
        
        logger.info(f"Input features for prediction: {list(input_filtered.columns)}")
        
        # Apply preprocessing
        X_processed = preprocessor.transform(input_filtered)
        logger.info(f"Processed input shape: {X_processed.shape}")
        
        # Convert to tensor and predict
        with torch.no_grad():
            inputs = torch.FloatTensor(X_processed)
            outputs = model(inputs)
        
        # Process outputs based on problem type
        if metadata['problem_type'] == 'binary_classification':
            # For binary classification, apply sigmoid and return probability
            probabilities = torch.sigmoid(outputs).squeeze()
            if len(probabilities.shape) == 0:  # Single prediction
                return float(probabilities)
            return probabilities.tolist()
            
        elif metadata['problem_type'] == 'multiclass_classification':
            # For multiclass, return class indices
            predicted_classes = outputs.argmax(dim=1)
            if len(predicted_classes) == 1:  # Single prediction
                return int(predicted_classes[0])
            return predicted_classes.tolist()
            
        else:  # regression
            predicted_values = outputs.squeeze()
            if len(predicted_values.shape) == 0:  # Single prediction
                return float(predicted_values)
            return predicted_values.tolist()
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Prediction error: {str(e)}")