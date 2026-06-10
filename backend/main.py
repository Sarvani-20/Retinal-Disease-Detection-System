from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
from infer import predict

# ✅ Create the app first
app = FastAPI()

# ✅ Setup folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("gradcam_outputs", exist_ok=True)

# ✅ Mount static folders
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/gradcam", StaticFiles(directory="gradcam_outputs"), name="gradcam")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define routes AFTER app is created
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    uid = str(uuid.uuid4())
    label, conf, original_url, heatmap_url = predict(file_path, uid)

    response = {
        "label": label,
        "confidence": float(conf),
        "original": original_url,
        "heatmap": heatmap_url
    }

    # Disease classification
    if "gla" in label.lower():
        response["glaucoma"] = {"present": True, "severity": label, "confidence": float(conf)}
    elif "amd" in label.lower():
        response["amd"] = {"present": True, "severity": label, "confidence": float(conf)}
    elif label.lower() == "healthy":
        response["healthy"] = {"present": True}
    else:
        response["dr"] = {"present": True, "severity": label, "confidence": float(conf)}

    if os.path.exists(file_path):
        os.remove(file_path)

    return response