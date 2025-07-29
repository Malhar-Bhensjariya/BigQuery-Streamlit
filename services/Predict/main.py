from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import logging
from typing import Dict, Any, Union, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("FastAPI app is starting up...")

app = FastAPI(
    title="ML Prediction Service",
    description="API for making predictions using trained models"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    bucket: str
    model_path: str
    data: Dict[str, Any]  # Single prediction input
    return_probs: bool = False  # For classification only

class BatchPredictRequest(BaseModel):
    bucket: str
    model_path: str
    data: List[Dict[str, Any]]  # Batch prediction input

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        logger.info(f"Prediction request for model: {req.model_path}")

        # Lazy import
        from predict import load_model, predict_input, validate_input

        model, preprocessor, metadata, feature_names = load_model(
            req.bucket, req.model_path
        )

        # Get original features that the model expects
        original_features = get_original_features_from_metadata(metadata)
        
        # Filter input data to only include original features
        filtered_data = {k: v for k, v in req.data.items() if k in original_features}
        input_df = pd.DataFrame([filtered_data])
        
        validate_input(input_df, original_features, metadata)

        prediction = predict_input(model, preprocessor, input_df, metadata)

        # Convert prediction back to original labels if classification
        if metadata.get('reverse_target_mapping') and metadata['problem_type'] != 'regression':
            if isinstance(prediction, list):
                prediction = [metadata['reverse_target_mapping'].get(pred, pred) for pred in prediction]
            else:
                prediction = metadata['reverse_target_mapping'].get(prediction, prediction)

        if req.return_probs and metadata['problem_type'] != 'regression':
            import torch
            with torch.no_grad():
                inputs = torch.FloatTensor(preprocessor.transform(input_df))
                outputs = model(inputs)
                if metadata['problem_type'] == 'binary_classification':
                    probs = torch.sigmoid(outputs).squeeze().item()
                else:
                    probs = torch.softmax(outputs, dim=1).squeeze().tolist()
            return {
                "status": "success",
                "prediction": prediction,
                "probabilities": probs
            }

        return {
            "status": "success",
            "prediction": prediction
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(req: BatchPredictRequest):
    try:
        logger.info(f"Batch prediction request for model: {req.model_path}")

        # Lazy import
        from predict import load_model, predict_input, validate_input

        model, preprocessor, metadata, feature_names = load_model(
            req.bucket, req.model_path
        )

        # Get original features that the model expects
        original_features = get_original_features_from_metadata(metadata)
        
        # Filter input data to only include original features
        filtered_data = []
        for row in req.data:
            filtered_row = {k: v for k, v in row.items() if k in original_features}
            filtered_data.append(filtered_row)
        
        input_df = pd.DataFrame(filtered_data)
        validate_input(input_df, original_features, metadata)

        predictions = predict_input(model, preprocessor, input_df, metadata)

        # Convert predictions back to original labels if classification
        if metadata.get('reverse_target_mapping') and metadata['problem_type'] != 'regression':
            if isinstance(predictions, list):
                predictions = [metadata['reverse_target_mapping'].get(pred, pred) for pred in predictions]

        return {
            "status": "success",
            "predictions": predictions
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

def get_original_features_from_metadata(metadata: Dict) -> List[str]:
    """Extract original feature names from metadata, excluding encoded features"""
    feature_metadata = metadata.get('feature_metadata', {})
    original_features = []
    
    for feature_name, feature_info in feature_metadata.items():
        # Skip encoded features (like cat__gender_Female, num__age, etc.)
        if feature_info.get('is_encoded', False):
            continue
        
        # Get the original feature name
        original_feature = feature_info.get('original_feature', feature_name)
        
        # Add to list if not already present
        if original_feature not in original_features:
            original_features.append(original_feature)
    
    return original_features

@app.get("/model_info")
async def get_model_info(bucket: str, model_path: str):
    try:
        # Lazy import
        from predict import load_model

        model, preprocessor, metadata, feature_names = load_model(bucket, model_path)

        return {
            "status": "success",
            "metadata": metadata,
            "features": get_original_features_from_metadata(metadata),
            "removed_columns": metadata.get('removed_columns', []),
            "categorical_values": metadata.get('categorical_values', {}),
            "numeric_stats": metadata.get('numeric_stats', {})
        }
    except Exception as e:
        logger.error(f"Model info error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/form_metadata")
async def get_form_metadata(bucket: str, model_path: str):
    try:
        logger.info(f"Fetching form metadata for model: {model_path}")
        from predict import load_model

        _, _, metadata, _ = load_model(bucket, model_path)
        
        # Extract form-ready metadata
        form_fields = {}
        
        # Get metadata
        feature_metadata = metadata.get('feature_metadata', {})
        categorical_values = metadata.get('categorical_values', {})
        numeric_stats = metadata.get('numeric_stats', {})
        
        # Create form fields from original features only
        processed_features = set()
        
        for feature_name, feature_info in feature_metadata.items():
            # Skip encoded features
            if feature_info.get('is_encoded', False):
                continue
                
            original_feature = feature_info.get('original_feature', feature_name)
            
            # Skip if already processed
            if original_feature in processed_features:
                continue
                
            processed_features.add(original_feature)
            
            if feature_info.get('type') == 'numeric':
                stats = numeric_stats.get(original_feature, {})
                form_fields[original_feature] = {
                    "type": "numeric",
                    "input_type": "number",
                    "required": True,
                    "min": stats.get('min'),
                    "max": stats.get('max'),
                    "placeholder": f"Enter {original_feature.replace('_', ' ').title()}",
                    "description": f"Range: {stats.get('min', 'N/A')} - {stats.get('max', 'N/A')}",
                    "default_value": stats.get('median')
                }
                
            elif feature_info.get('type') == 'categorical':
                categories = categorical_values.get(original_feature, feature_info.get('categories', []))
                form_fields[original_feature] = {
                    "type": "categorical", 
                    "input_type": "select",
                    "required": True,
                    "options": categories,
                    "placeholder": f"Select {original_feature.replace('_', ' ').title()}",
                    "default_value": categories[0] if categories else None
                }

        return {
            "status": "success",
            "problem_type": metadata.get('problem_type'),
            "target_column": metadata.get('target_column'),
            "form_fields": form_fields,
            "model_info": {
                "training_date": metadata.get('training_date'),
                "removed_columns": metadata.get('removed_columns', []),
                "metrics": metadata.get('metrics', {})
            }
        }

    except Exception as e:
        logger.error(f"Form metadata error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Metadata generation failed: {str(e)}"
            }
        )

@app.get("/prediction_schema")
async def get_prediction_schema(bucket: str, model_path: str):
    """
    Get the schema for making predictions - useful for API documentation
    """
    try:
        from predict import load_model

        _, _, metadata, _ = load_model(bucket, model_path)
        
        schema = {
            "required_fields": [],
            "field_types": {},
            "field_constraints": {}
        }
        
        feature_metadata = metadata.get('feature_metadata', {})
        categorical_values = metadata.get('categorical_values', {})
        numeric_stats = metadata.get('numeric_stats', {})
        
        # Get original features only (not encoded ones)
        processed_features = set()
        
        for feature_name, feature_info in feature_metadata.items():
            # Skip encoded features
            if feature_info.get('is_encoded', False):
                continue
                
            original_feature = feature_info.get('original_feature', feature_name)
            
            # Skip if already processed
            if original_feature in processed_features:
                continue
                
            processed_features.add(original_feature)
            schema["required_fields"].append(original_feature)
            
            if feature_info.get('type') == 'numeric':
                schema["field_types"][original_feature] = "number"
                stats = numeric_stats.get(original_feature, {})
                schema["field_constraints"][original_feature] = {
                    "min": stats.get('min'),
                    "max": stats.get('max')
                }
            elif feature_info.get('type') == 'categorical':
                schema["field_types"][original_feature] = "string"
                categories = categorical_values.get(original_feature, feature_info.get('categories', []))
                schema["field_constraints"][original_feature] = {
                    "enum": categories
                }
        
        # Create example request with only original features
        example_request = {}
        for field in schema["required_fields"]:
            if schema["field_types"][field] == "string":
                # Use first category for categorical fields
                categories = schema["field_constraints"][field].get("enum", [])
                example_request[field] = categories[0] if categories else "example_value"
            else:
                # Use median for numeric fields
                example_request[field] = numeric_stats.get(field, {}).get('median', 0)
        
        return {
            "status": "success",
            "schema": schema,
            "example_request": example_request
        }
        
    except Exception as e:
        logger.error(f"Schema generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )