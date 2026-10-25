from contextlib import asynccontextmanager
import uuid
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from Utils.loader import load_all_models
from Utils.classifier import bytes_to_tensor, run_inference
from Utils.attacker import apply_attack
from Utils.mda import generate_mda_heatmap

models = {}
heatmap_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    models.update(load_all_models())
    print("All models loaded successfully.")
    yield
    models.clear()
    print("Models cleared.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


def _generate_heatmap_task(task_id: str, model_name: str):
    """Background task to generate heatmap asynchronously."""
    try:
        tensor = heatmap_store[task_id]["tensor"]
        selected_model = models[model_name]
        heatmap = generate_mda_heatmap(tensor, selected_model)
        heatmap_store[task_id]["heatmap_base64"] = heatmap
        heatmap_store[task_id]["tensor"] = None  # Free GPU memory
        heatmap_store[task_id]["status"] = "ready"
    except Exception as exc:
        heatmap_store[task_id]["status"] = "error"
        heatmap_store[task_id]["error"] = str(exc)


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    model: str = "standard",
    attack: bool = False,
    epsilon: float = 0.01,
    background_tasks: BackgroundTasks = None,
):
    if model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not available")

    image_bytes = await file.read()
    tensor = bytes_to_tensor(image_bytes)
    selected_model = models[model]
    standard_model = models["standard"]

    if attack:
        tensor = apply_attack(tensor, "fgsm", standard_model, epsilon)

    result = run_inference(tensor, selected_model)

    heatmap_id = uuid.uuid4().hex
    heatmap_store[heatmap_id] = {"status": "pending", "heatmap_base64": None, "error": None, "tensor": tensor}
    background_tasks.add_task(_generate_heatmap_task, heatmap_id, model)

    return {
        "predicted_class":   result["predicted_class"],
        "confidence_scores": result["confidence_scores"],
        "heatmap_id":        heatmap_id,
        "heatmap_status":    "pending",
    }


@app.get("/heatmap/{heatmap_id}")
def get_heatmap(heatmap_id: str):
    """Retrieve heatmap status and data. Polling this endpoint tracks heatmap generation progress."""
    data = heatmap_store.get(heatmap_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Heatmap ID not found")

    if data["status"] == "pending":
        return {"status": "pending"}
    if data["status"] == "ready":
        return {"status": "ready", "heatmap_base64": data["heatmap_base64"]}
    return {"status": "error", "error": data["error"]}