"""FastAPI application for tiled image classification.

Upload a multi-channel image (or multiple single-channel files) and get:
  - Per-tile predictions
  - Whole-image prediction via majority vote across tiles

Models are loaded from MLflow (registered model name, run name, or checkpoint).
"""
import io
import sys
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from api.config import Settings
from api.predictor import TilePredictor
from database.dblogger import DBLogger
from utils.filename_parser import extract_info_from_filename
import asyncio
from starlette.concurrency import run_in_threadpool

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Global predictor instance (loaded at startup)
predictor: TilePredictor | None = None
db_logger: DBLogger | None = None
settings = Settings()
inference_sem = asyncio.Semaphore(2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup from MLflow."""
    global predictor
    global db_logger 
    
    if not settings.has_model_source:
        print("WARNING: No model source configured.")
        print("Set one of: API_MODEL_NAME or API_RUN_NAME")
        print("The /predict endpoint will return an error until a model is loaded.")
    else:
        source = settings.model_name or settings.run_name
        print(f"Loading model: {source}")

        if settings.has_db_uri:
            db_logger = DBLogger(db_uri=settings.db_uri)
            print(f"Connecting to database: {settings.db_uri}")
            try:
                db_logger.connect()
            except Exception as e:  # noqa: BLE001
                print(f"Failed to connect to database: {e}")
                db_logger = None
        else:
            print("WARNING: No database URI configured.")
            print("Set API_DB_URI to enable database logging.")
            db_logger = None

        predictor = TilePredictor(
            tracking_uri=settings.tracking_uri,
            experiment_name=settings.experiment_name,
            model_name=settings.model_name,
            run_name=settings.run_name,
            crop_size=settings.crop_size,
            stride=settings.effective_stride,
            device=settings.device,
            db_logger=db_logger,
            artifact_root=settings.artifact_root
        )
        print(f"Model loaded. Source: {predictor.model_info['source']}")
        print(f"  Classes: {predictor.class_names}")
        print(f"  Tile size: {predictor.crop_size}, Stride: {predictor.stride}")
        print(f"  In channels: {predictor.in_channels}")
    
    yield
    
    # Cleanup
    predictor = None
    if db_logger:
        db_logger.close()


app = FastAPI(
    title="Image Classifier API",
    description="Tiled multi-channel image classification with majority voting. "
                "Models loaded from MLflow registry, run name, or checkpoint.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "device": settings.device,
        "database_connected": db_logger is not None,
    }


@app.get("/model")
async def model_info():
    """Return detailed info about the loaded model and its config."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    
    return {
        "source": predictor.model_info.get("source"),
        "model_class": predictor.model_info.get("model_class"),
        "backbone": predictor.model_info.get("backbone"),
        "run_id": predictor.model_info.get("run_id"),
        "num_classes": predictor.model_info.get("num_classes"),
        "class_names": predictor.class_names,
        "in_channels": predictor.in_channels,
        "crop_size": predictor.crop_size,
        "stride": predictor.stride,
        "device": str(predictor.device),
    }
@app.get("/db")
async def db_info():
    """Return detailed info about the database connection."""
    if db_logger is None:
        raise HTTPException(status_code=503, detail="No database connection.")
    
    return {
        "connected": True,
        "uri": settings.db_uri,
    }

@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(..., description="Image files (one per channel, ordered C1..CN)"),  # noqa: B008
    root_path: str = Form(..., description="Root path for the image"),
    crop_size: int | None = Query(None, description="Override tile crop size"),
    stride: int | None = Query(None, description="Override tile stride"),
):
    """Predict on an uploaded image.

    Accepts multiple image files (one per channel), which are stacked in
    ascending channel-number order to match training.

    Returns per-tile predictions and a majority-vote whole-image prediction.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Set API_MODEL_NAME or API_RUN_NAME.",
        )
    
    try:
        image_metadata,image_channels = await _load_image_from_uploads(files)
        image_metadata["root_path"] = root_path
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
    try:
        async with inference_sem:
            result = await run_in_threadpool(predictor.predict, image_channels, image_metadata, crop_size, stride)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return JSONResponse(content=result)

def extract_infos(filenames : list[str]) -> dict:
    """Extract information from filenames.
    
    - filenames: list of filenames
    """
    infos = []
    for filename in filenames:
        info = extract_info_from_filename(filename)
        infos.append(info)
    
    ## normalize metdata shape like training metdata
    ## all plates , well, and fields should be the same for one sample
    assert len({info["plate"] for info in infos}) == 1, "All plates should be the same for one sample"
    assert len({info["well"] for info in infos}) == 1, "All wells should be the same for one sample"
    assert len({info["field"] for info in infos}) == 1, "All fields should be the same for one sample"



    infos_dict = {
        "plate": infos[0]["plate"],
        "well": infos[0]["well"],
        "field": infos[0]["field"],
        "channel_files": [info["filename"] for info in infos],
        "channels" : [info["channel"] for info in infos]
    }
    return infos_dict


async def _load_image_from_uploads(files: list[UploadFile]) -> tuple[dict, list[np.ndarray]]:
    """Load uploaded files into a list of per-channel (H, W) numpy arrays.

    - One or more image files (.tif, .jxl, etc.): each file = one channel,
      returned in upload order. Channel-axis canonicalization to ascending
      channel number happens later, inside TilePredictor.predict via
      chans_reorder, so this function stays a pure loader.
    """
    file_names = []
    # Multiple files: each file = one channel
    channels = []
    for f in files:
        content = await f.read()
        img = _load_single_image(content, f.filename or "upload")
        channels.append(img)
        file_names.append(f.filename or "upload")
        
    shapes = [ch.shape for ch in channels]
    if len(set(shapes)) > 1:
        raise ValueError(f"All channel images must have same dimensions. Got: {shapes}")
    assert len(file_names) == len(channels)

    ## extract infos from filenames
    infos = extract_infos(file_names)
    infos["shape"] = channels[0].shape

    return infos, channels
    
def _load_single_image(content: bytes, filename: str) -> np.ndarray:
    """Load a single image from bytes into a 2D numpy array."""
    import cv2
    import pillow_jxl  # noqa: F401  register JXL support with PIL
    from PIL import Image

    suffix = Path(filename).suffix.lower()
    if suffix in (".tif", ".tiff"):
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, -1)
    elif suffix == ".jxl":
        img = np.array(Image.open(io.BytesIO(content)))
    else:
        buf = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(buf, -1)
 
    if img is None:
        raise ValueError(f"Failed to decode image: {filename}")
 
    return img.astype(np.float32)



