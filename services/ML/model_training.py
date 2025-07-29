import os
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from google.cloud import bigquery, storage
from model_definition import PyTorchModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training pipeline from BigQuery data to GCS model storage"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.bq_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        
        # Initialize instance variables
        self.problem_type = None
        self.num_classes = None
        
        # Configuration defaults
        self.config = {
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
    
    def load_data_from_bq(self, dataset_id: str, table_id: str) -> pd.DataFrame:
        """Load data from BigQuery table"""
        query = f"""
            SELECT * 
            FROM `{self.project_id}.{dataset_id}.{table_id}`
        """
        logger.info(f"Loading data from {dataset_id}.{table_id}")
        return self.bq_client.query(query).to_dataframe()
    
    def determine_target_column(self, data: pd.DataFrame) -> Tuple[str, bool]:
        """
        Find the best candidate target column with priority order
        Returns (column_name, is_fallback_selection)
        """
        # Get column names and types
        clean_fields = data.columns.tolist()
        type_report = {}
        
        # Analyze column types
        for col in clean_fields:
            unique_count = data[col].nunique()
            dtype = data[col].dtype
            
            # Determine column type similar to CSV processing
            if dtype == 'bool' or (unique_count <= 2 and data[col].isin([0, 1, True, False, 'true', 'false', 'yes', 'no']).all()):
                type_report[col] = 'BOOLEAN'
            elif dtype in ['int64', 'int32']:
                type_report[col] = 'INT64'
            elif dtype in ['float64', 'float32']:
                type_report[col] = 'FLOAT64'
            else:
                type_report[col] = 'STRING'
        
        # Priority 1: Common target names
        COMMON_TARGETS = {'target', 'label', 'class', 'outcome', 'churn', 'result', 'score', 'y'}
        for col in clean_fields:
            if col.lower() in COMMON_TARGETS:
                logger.info(f"Found common target column: {col}")
                return col, False
        
        # Priority 2: Boolean columns
        for col, col_type in type_report.items():
            if col_type == 'BOOLEAN':
                logger.info(f"Selected boolean column as target: {col}")
                return col, True
        
        # Priority 3: Numeric columns with reasonable number of unique values for classification
        numeric_cols = [col for col, typ in type_report.items() 
                       if typ in ('INT64', 'FLOAT64')]
        
        for col in numeric_cols:
            unique_count = data[col].nunique()
            # Good for classification if 2-50 unique values
            if 2 <= unique_count <= 50:
                logger.info(f"Selected numeric column with {unique_count} unique values as target: {col}")
                return col, True
        
        # Priority 4: Any numeric column (for regression)
        if numeric_cols:
            col = numeric_cols[0]
            logger.info(f"Selected first numeric column as regression target: {col}")
            return col, True
        
        # Priority 5: Last resort - use last column
        if clean_fields:
            col = clean_fields[-1]
            logger.warning(f"Using last column as fallback target: {col}")
            return col, True
        
        raise ValueError("No suitable target column found in the dataset")
    
    def determine_problem_type(self, target_column: str, data: pd.DataFrame) -> str:
        """Determine if problem is classification or regression"""
        target_unique = data[target_column].nunique()
        
        # Check for null values in target
        if data[target_column].isnull().any():
            logger.warning(f"Target column '{target_column}' contains null values")
        
        # Binary classification
        if target_unique == 2:
            return 'binary_classification'
        # Multi-class classification (increased threshold for better heuristic)
        elif target_unique > 2 and target_unique <= 50 and data[target_column].dtype in ['object', 'category', 'int64']:
            return 'multiclass_classification'
        # Regression
        else:
            return 'regression'
    
    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        problem_type: str
    ) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
        logger.info("Starting data preprocessing")
        
        # Remove ID-like columns before processing
        columns_to_remove = []
        for col in data.columns:
            if col == target_column:
                continue
                
            col_lower = col.lower()
            # Check for common ID patterns
            if (col_lower.endswith('id') or col_lower.startswith('id') or 
                col_lower.endswith('_id') or col_lower == 'index' or
                col_lower.endswith('number') or col_lower.endswith('code')):
                columns_to_remove.append(col)
                continue
                
            # Check if column has all unique values (likely an ID)
            if data[col].nunique() == len(data) and data[col].nunique() > 10:
                columns_to_remove.append(col)
                continue
                
            # Check correlation with index for numeric columns
            if data[col].dtype in ['int64', 'float64']:
                try:
                    correlation = data[col].corr(pd.Series(range(len(data))))
                    if abs(correlation) > 0.95:
                        columns_to_remove.append(col)
                except:
                    pass
        
        if columns_to_remove:
            logger.info(f"Removing ID-like columns: {columns_to_remove}")
            data = data.drop(columns=columns_to_remove)
        
        # Clean data and separate features/target
        data_clean = data.dropna(subset=[target_column])
        X = data_clean.drop(columns=[target_column])
        y = data_clean[target_column].copy()
        
        # Store target mapping for classification
        target_mapping = None
        reverse_target_mapping = None
        if problem_type in ['binary_classification', 'multiclass_classification']:
            unique_values = sorted(y.unique())
            target_mapping = {val: idx for idx, val in enumerate(unique_values)}
            reverse_target_mapping = {idx: val for val, idx in target_mapping.items()}
            y = y.map(target_mapping)
            self.num_classes = len(unique_values)
        
        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get categorical unique values
        categorical_values = {}
        for feature in categorical_features:
            unique_vals = X[feature].dropna().unique()
            categorical_values[feature] = sorted([str(val) for val in unique_vals])
        
        # Get numeric statistics
        numeric_stats = {}
        for feature in numeric_features:
            stats = X[feature].describe()
            numeric_stats[feature] = {
                'min': float(stats['min']),
                'max': float(stats['max']),
                'mean': float(stats['mean']),
                'median': float(stats['50%'])
            }
        
        # Initialize metadata storage
        feature_metadata = {}
        transformers = []
        
        # Numeric pipeline
        if numeric_features:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
            
            for feature in numeric_features:
                feature_metadata[feature] = {
                    "original_feature": feature,
                    "type": "numeric",
                    "transformer": "numeric",
                    "imputation": "median",
                    "scaling": "standard"
                }
        
        # Categorical pipeline
        if categorical_features:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
            
            for feature in categorical_features:
                feature_metadata[feature] = {
                    "original_feature": feature,
                    "type": "categorical",
                    "transformer": "categorical",
                    "imputation": "most_frequent",
                    "encoding": "one-hot",
                    "categories": categorical_values[feature]
                }
        
        # Create and fit preprocessor
        preprocessor = ColumnTransformer(transformers)
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names and update metadata for encoded features
        feature_names = preprocessor.get_feature_names_out()
        
        # Add metadata for encoded categorical features
        if categorical_features:
            encoder = None
            for name, transformer, features in preprocessor.transformers_:
                if name == 'cat':
                    encoder = transformer.named_steps['encoder']
                    break
            
            if encoder:
                encoded_features = encoder.get_feature_names_out(categorical_features)
                for encoded_feature in encoded_features:
                    parts = encoded_feature.split('_', 1)
                    if len(parts) == 2:
                        original_feature, category = parts
                        feature_metadata[encoded_feature] = {
                            "original_feature": original_feature,
                            "type": "categorical_encoded",
                            "transformer": "categorical",
                            "is_encoded": True,
                            "category_value": category
                        }
        
        # Store comprehensive metadata in preprocessor
        preprocessor.removed_columns_ = columns_to_remove
        preprocessor.feature_metadata_ = feature_metadata
        preprocessor.categorical_values_ = categorical_values
        preprocessor.numeric_stats_ = numeric_stats
        preprocessor.target_mapping_ = target_mapping
        preprocessor.reverse_target_mapping_ = reverse_target_mapping
        
        return pd.DataFrame(X_processed, columns=feature_names), y, preprocessor
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets"""
        # Determine stratification
        stratify_train_test = y if self.problem_type in ['binary_classification', 'multiclass_classification'] else None
        
        # First split to separate out test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=stratify_train_test
        )
        
        # Determine stratification for validation split
        stratify_train_val = y_train_val if self.problem_type in ['binary_classification', 'multiclass_classification'] else None
        
        # Second split to create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config['val_size'] / (1 - self.config['test_size']),  # Adjust for remaining data
            random_state=self.config['random_state'],
            stratify=stratify_train_val
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Convert pandas DataFrames/Series to PyTorch DataLoaders"""
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train.values)
        X_val_t = torch.FloatTensor(X_val.values)
        X_test_t = torch.FloatTensor(X_test.values)
        
        if self.problem_type == 'multiclass_classification':
            y_train_t = torch.LongTensor(y_train.values)
            y_val_t = torch.LongTensor(y_val.values)
            y_test_t = torch.LongTensor(y_test.values)
        else:
            y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
            y_val_t = torch.FloatTensor(y_val.values).unsqueeze(1)
            y_test_t = torch.FloatTensor(y_test.values).unsqueeze(1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, input_size):
        """Train the PyTorch model"""
        logger.info("Starting model training")
        
        # Initialize model
        model = PyTorchModel(
            input_size=input_size,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hidden_layers=self.config['model_params']['hidden_layers'],
            dropout_rate=self.config['model_params']['dropout_rate'],
            l2_reg=self.config['model_params']['l2_reg']
        )
        
        # Define loss and optimizer
        if self.problem_type == 'binary_classification':
            criterion = nn.BCELoss()
        elif self.problem_type == 'multiclass_classification':
            criterion = nn.CrossEntropyLoss()
        else:  # regression
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=self.config['model_params']['learning_rate'])
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Adjust outputs for binary classification
                if self.problem_type == 'binary_classification':
                    outputs = outputs.squeeze()
                    labels = labels.squeeze().float()
                
                loss = criterion(outputs, labels) + model.l2_loss()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    
                    # Adjust outputs for binary classification
                    if self.problem_type == 'binary_classification':
                        outputs = outputs.squeeze()
                        labels = labels.squeeze().float()
                    
                    loss = criterion(outputs, labels) + model.l2_loss()
                    val_loss += loss.item() * inputs.size(0)
            
            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f'Epoch {epoch+1}/{self.config["epochs"]} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model performance on test set"""
        logger.info("Evaluating model performance")
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                all_preds.append(outputs)
                all_labels.append(labels)
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if self.problem_type == 'binary_classification':
            y_pred = all_preds.squeeze().numpy()
            y_test = all_labels.squeeze().numpy()
            y_pred_class = (y_pred > 0.5).astype(int)
            
            return {
                'accuracy': accuracy_score(y_test, y_pred_class),
                'precision': precision_score(y_test, y_pred_class, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_class, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_class, average='weighted', zero_division=0)
            }
        else:  # regression
            y_pred = all_preds.squeeze().numpy()
            y_test = all_labels.squeeze().numpy()
            
            return {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
    
    def save_model(
        self,
        model: nn.Module,
        preprocessor: ColumnTransformer,
        dataset_id: str,
        table_id: str,
        target_column: str,
        metrics: Dict[str, float]
    ) -> str:
        try:
            # Get all metadata from preprocessor
            feature_metadata = getattr(preprocessor, 'feature_metadata_', {})
            feature_names = preprocessor.get_feature_names_out()
            removed_columns = getattr(preprocessor, 'removed_columns_', [])
            categorical_values = getattr(preprocessor, 'categorical_values_', {})
            numeric_stats = getattr(preprocessor, 'numeric_stats_', {})
            target_mapping = getattr(preprocessor, 'target_mapping_', None)
            reverse_target_mapping = getattr(preprocessor, 'reverse_target_mapping_', None)
            
            # Create model package with enhanced metadata
            model_package = {
                'model_state': model.state_dict(),
                'model_class': 'model_definition.PyTorchModel',
                'model_params': {
                    'input_size': len(feature_names),
                    'problem_type': self.problem_type,
                    'num_classes': self.num_classes,
                    'hidden_layers': self.config['model_params']['hidden_layers'],
                    'dropout_rate': self.config['model_params']['dropout_rate'],
                    'l2_reg': self.config['model_params']['l2_reg']
                },
                'preprocessor': preprocessor,
                'metadata': {
                    'dataset_id': dataset_id,
                    'table_id': table_id,
                    'target_column': target_column,
                    'problem_type': self.problem_type,
                    'training_date': datetime.now().isoformat(),
                    'feature_metadata': feature_metadata,
                    'feature_names': feature_names.tolist(),
                    'removed_columns': removed_columns,
                    'categorical_values': categorical_values,
                    'numeric_stats': numeric_stats,
                    'target_mapping': target_mapping,
                    'reverse_target_mapping': reverse_target_mapping,
                    'metrics': metrics,
                    'config': self.config
                }
            }
            
            # Save to GCS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"models/{dataset_id}/{table_id}/{target_column}.pkl"
            
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(model_filename)
            blob.upload_from_string(pickle.dumps(model_package))
            
            logger.info(f"Model saved to gs://{self.bucket_name}/{model_filename}")
            return model_filename
            
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model save failed: {str(e)}")
        
    def train_pipeline(
        self,
        dataset_id: str,
        table_id: str,
        target_column: str = None,
        config: Dict = None
    ) -> Dict:
        """Complete training pipeline from data loading to model saving"""
        try:
            # Update config if provided
            if config:
                self.config.update(config)
            
            logger.info(f"Starting training pipeline for {dataset_id}.{table_id}, target: {target_column}")
            
            # Load data
            data = self.load_data_from_bq(dataset_id, table_id)
            
            if data.empty:
                raise ValueError("No data loaded from BigQuery")
            
            # Auto-detect target if not provided
            if target_column is None:
                target_column, is_fallback = self.determine_target_column(data)
                if is_fallback:
                    logger.warning(f"Auto-detected target column: '{target_column}' (fallback selection - please verify)")
                else:
                    logger.info(f"Auto-detected target column: '{target_column}' (high confidence)")
            else:
                # Validate provided target column exists
                if target_column not in data.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                logger.info(f"Using provided target column: '{target_column}'")
            
            
            # Determine problem type
            self.problem_type = self.determine_problem_type(target_column, data)
            if self.problem_type in ['binary_classification', 'multiclass_classification']:
                self.num_classes = data[target_column].nunique()
            logger.info(f"Problem type: {self.problem_type}, Classes: {self.num_classes}")
            
            # Preprocess data
            X, y, preprocessor = self.preprocess_data(data, target_column, self.problem_type)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Create DataLoaders
            train_loader, val_loader, test_loader = self.create_dataloaders(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            
            # Train model
            input_size = X_train.shape[1]
            logger.info(f"Input size: {input_size}")
            model = self.train_model(train_loader, val_loader, input_size)
            
            # Evaluate model
            metrics = self.evaluate_model(model, test_loader)
            logger.info(f"Model metrics: {metrics}")
            
            # Save model
            model_path = self.save_model(
                model, preprocessor,
                dataset_id, table_id, target_column, metrics
            )
            
            return {
                'status': 'success',
                'model_path': model_path,
                'metrics': metrics,
                'problem_type': self.problem_type,
                'target_column': target_column,
                'num_classes': self.num_classes,
                'input_size': input_size
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
