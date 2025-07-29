from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from model_training import ModelTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Request model
class TrainRequest(BaseModel):
    project_id: str
    bucket_name: str
    dataset_id: str
    table_id: str
    target_column: str | None = None
    config: dict | None = None

@app.post("/train")
async def train_model(data: TrainRequest):
    try:
        target_info = f"with target '{data.target_column}'" if data.target_column else "with auto-detected target"
        logger.info(f"Starting training for {data.dataset_id}.{data.table_id} {target_info}")

        trainer = ModelTrainer(
            project_id=data.project_id,
            bucket_name=data.bucket_name
        )

        result = trainer.train_pipeline(
            dataset_id=data.dataset_id,
            table_id=data.table_id,
            target_column=data.target_column,
            config=data.config
        )

        if result['status'] == 'success':
            logger.info(f"Training successful: {result['model_path']}")
            return {
                "status": "success",
                "model_path": result["model_path"],
                "metrics": result["metrics"],
                "problem_type": result["problem_type"],
                "target_column": result["target_column"],
                "num_classes": result.get("num_classes"),
                "input_size": result["input_size"]
            }
        else:
            logger.error(f"Training failed: {result['message']}")
            raise HTTPException(status_code=400, detail=result["message"])

    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )
