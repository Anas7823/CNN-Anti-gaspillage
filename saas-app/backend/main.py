import json
import os
import io
import numpy as np
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

import tensorflow as tf

# ─── Paths (relatives à ce fichier) ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # racine du projet CNN
MODEL_PATH = BASE_DIR / "best_model" / "fruit_quality_mobilenet_fp32.tflite"
KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"
PLOTS_DIR = BASE_DIR / "plots"

# ─── Vérifications au démarrage ───────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle TFLite introuvable : {MODEL_PATH}")
if not KB_PATH.exists():
    raise FileNotFoundError(f"Base de connaissances introuvable : {KB_PATH}")

# ─── Classes dans l'ordre alphabétique Keras ──────────────────────────────────
CLASS_NAMES = [
    "apple_fresh", "apple_rotten",
    "banana_fresh", "banana_rotten",
    "bellpepper_fresh", "bellpepper_rotten",
    "carrot_fresh", "carrot_rotten",
    "cucumber_fresh", "cucumber_rotten",
    "grape_fresh", "grape_rotten",
    "guava_fresh", "guava_rotten",
    "jujube_fresh", "jujube_rotten",
    "mango_fresh", "mango_rotten",
    "orange_fresh", "orange_rotten",
    "pomegranate_fresh", "pomegranate_rotten",
    "potato_fresh", "potato_rotten",
    "strawberry_fresh", "strawberry_rotten",
    "tomato_fresh", "tomato_rotten",
]

# ─── Chargement du modèle TFLite (une seule fois au démarrage) ────────────────
print(f"[INFO] Chargement du modèle : {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = (160, 160)
print("[INFO] Modèle TFLite chargé avec succès.")

# ─── Base de connaissances ────────────────────────────────────────────────────
with open(KB_PATH, "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

# ─── Application FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="FreshScan AI — API Anti-Gaspillage",
    description="API de détection de qualité des fruits et légumes via CNN MobileNetV2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les images de courbes d'entraînement pour la page Portfolio
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
    print(f"[INFO] Dossier /plots monté depuis : {PLOTS_DIR}")
else:
    print(f"[WARNING] Dossier plots non trouvé : {PLOTS_DIR}")


# ─── Utilitaires ──────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Charge une image depuis les bytes, la redimensionne en 160×160
    et applique le prétraitement MobileNetV2 (normalisation entre -1 et 1).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    # tf.keras.applications.mobilenet_v2.preprocess_input : (x / 127.5) - 1
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)  # (1, 160, 160, 3)


def run_inference(img_array: np.ndarray) -> np.ndarray:
    """Lance l'inférence TFLite et retourne le vecteur de probabilités."""
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]  # shape (28,)


def build_response(raw_class: str, confidence: float) -> dict:
    """Construit la réponse JSON enrichie avec les suggestions."""
    parts = raw_class.split("_", 1)  # ex: "bellpepper_fresh" → ["bellpepper", "fresh"]
    fruit_key = parts[0]
    status = parts[1] if len(parts) > 1 else "fresh"

    kb_entry = KNOWLEDGE_BASE.get(fruit_key, {})
    name_fr = kb_entry.get("name_fr", fruit_key.capitalize())
    emoji = kb_entry.get("emoji", "🍽️")
    status_fr = "Frais" if status == "fresh" else "Avarié / Abîmé"
    label_fr = f"{name_fr} {status_fr}"

    suggestions_data = kb_entry.get(status, {})
    tips = suggestions_data.get("tips", ["Consultez une recette anti-gaspillage adaptée."])
    suggestion_title = (
        "💡 Astuces de Conservation" if status == "fresh" else "♻️ Recettes Anti-Gaspillage"
    )

    return {
        "raw_class": raw_class,
        "fruit_key": fruit_key,
        "status": status,
        "label_fr": label_fr,
        "emoji": emoji,
        "confidence": round(float(confidence), 4),
        "suggestions": {
            "title": suggestion_title,
            "tips": tips,
        },
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def health_check():
    return {"status": "ok", "model": "MobileNetV2 TFLite FP32", "classes": len(CLASS_NAMES)}


@app.get("/api/classes", summary="Liste des 28 classes supportées")
def get_classes():
    return {"classes": CLASS_NAMES, "total": len(CLASS_NAMES)}


@app.post("/api/predict", summary="Analyser un fruit ou légume")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit une image (jpg, png, webp…), effectue l'inférence avec le modèle
    MobileNetV2 TFLite et renvoie la classe détectée avec les suggestions associées.
    """
    # Validation du type de fichier
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Utilisez JPG, PNG ou WEBP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10 Mo max
        raise HTTPException(status_code=413, detail="Image trop volumineuse (max 10 Mo).")

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Impossible de lire l'image : {e}")

    probabilities = run_inference(img_array)
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    raw_class = CLASS_NAMES[predicted_idx]

    return JSONResponse(content=build_response(raw_class, confidence))


@app.get("/api/plots", summary="Liste des fichiers de courbes disponibles")
def list_plots():
    """Retourne la liste des images de courbes disponibles dans /plots."""
    if not PLOTS_DIR.exists():
        return {"plots": [], "message": "Dossier plots introuvable."}

    png_files = [f.name for f in PLOTS_DIR.glob("*.png")]
    return {
        "plots": [{"filename": f, "url": f"/plots/{f}"} for f in sorted(png_files)],
        "total": len(png_files),
    }
