# BigQuery-Streamlit

```text
📦bigquery-streamlit/
├── 📂.streamlit/
│   └── secrets.toml          # GCP credentials and config
│
├── 📂components/
│   ├── __init__.py           # Makes folder a Python package
│   ├── gcp_clients.py        # GCP service clients (BigQuery, Storage)
│   ├── navigation.py         # State management and routing
│   └── vertex_ai.py          # Future Vertex AI integration (Empty for now)
│
├── 📂pages/
│   ├── __init__.py           # Makes folder a Python package
│   ├── Dataset_Selection.py
│   ├── Table_Selection.py
│   ├── File_Upload.py
│   └── Prediction.py          # Future prediction page (Empty for now)
│
├── 📜app.py                  # Main application entry point
├── 📜requirements.txt        # Python dependencies
├── 📜.gitignore              # Github ignore files/folders
└── 📜README.md               # Project documentation
```