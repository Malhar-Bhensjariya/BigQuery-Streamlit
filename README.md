# BigQuery-Streamlit

```text
📦bigquery-streamlit/
├── 📂frontend/                       
│   ├── 📂public/                      # React public assets
│   ├── 📂src/
│   │   ├── 📂components/ 
│   │   │   ├── PredictionForm.jsx
│   │   │   └── TrainModel.jsx
│   │   │
│   │   ├── 📂context/
│   │   │   └── AppContext.jsx
│   │   │
│   │   ├── 📂pages/                  
│   │   │   ├── DatasetSelection.jsx
│   │   │   ├── TableSelection.jsx
│   │   │   ├── FileUpload.jsx
│   │   │   └── Prediction.jsx
│   │   │
│   │   ├── App.jsx                   # Main app entry
│   │   └── main.jsx                  # React DOM render
│   │
│   ├── .env                          # Replaces secrets.toml (GCP creds + config)
│   ├── package.json                  # React dependencies
│   └── vite.config.js                # If using Vite (else use CRA config)
│
├── 📂prototype/
│   ├── 📂.streamlit/
│   │   └── secrets.toml          # GCP credentials and config
│   │
│   ├── 📂components/
│   │   ├── __init__.py           # Makes folder a Python package
│   │   ├── gcp_clients.py        # GCP service clients (BigQuery, Storage)
│   │   ├── navigation.py         # State management and routing
│   │   ├── prediction_form.py    # Prediction form for user input
│   │   └── train_model.py        # Training new model, retrain model functionality
│   │
│   ├── 📂pages/
│   │   ├── __init__.py           
│   │   ├── Dataset_Selection.py
│   │   ├── Table_Selection.py
│   │   ├── File_Upload.py
│   │   └── Prediction.py          
│   │
│   ├── app.py                  # Main application entry point
│   └── requirements.txt        # Python dependencies
│
├── 📂services/
│   ├── 📂Cloud_functions/
│   │   ├── 📂gcs_to_bq/
│   │   │   ├── 📂BQ_SQL/
│   │   │   │   ├── __init__.py          
│   │   │   │   ├── backup_manager.py                
│   │   │   │   ├── data_cleaner.py                  
│   │   │   │   ├── data_quality_analyzer.py   
│   │   │   │   └── validator.py 
│   │   │   ├── main.py 
│   │   │   └── Dockerfile               
│   ├── 📂ML/
│   │   ├── main.py          
│   │   ├── model_train.py          
│   │   ├── model_definition.py                  
│   │   ├── requirements.txt   
│   │   └── Dockerfile            # Exposed on Port 81       
│   ├── 📂Predict/
│   │   ├── main.py          
│   │   ├── predict.py          
│   │   ├── model_definition.py                
│   │   ├── requirements.txt   
│   │   └── Dockerfile            # Exposed on Port 81 
│   └──💲setup.sh                  # Main application entry point
│
├── .gitignore              # Github ignore files/folders
└── ℹ️README.md               # Project documentation
```