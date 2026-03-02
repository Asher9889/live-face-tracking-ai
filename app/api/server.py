from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np

from app.ai.insight_detector import InsightFaceEngine

app = FastAPI()
insight_engine = InsightFaceEngine()


@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/register-face")
async def register_face(files: List[UploadFile] = File(...)):

    raw_embeddings = []
    faces_meta = []

    for file in files:

        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            continue

        faces = insight_engine.detect_and_generate_embedding(img)

        if not faces:
            continue

        # choose best face (highest detection score)
        best_face = max(faces, key=lambda f: f["score"])

        embedding = best_face["embedding"]

        raw_embeddings.append(embedding)

        faces_meta.append({
            "bbox": best_face["bbox"].tolist(),
            "score": float(best_face["score"]),
            "yaw": float(best_face["pose"][0]),
            "pitch": float(best_face["pose"][1]),
            "roll": float(best_face["pose"][2]),
            "age": best_face["age"],
            "gender": best_face["gender"]
        })

    if len(raw_embeddings) == 0:
        return {
            "success": False,
            "message": "No valid faces detected"
        }

    # Convert list → numpy
    emb_matrix = np.stack(raw_embeddings)

    # mean embedding
    mean_embedding = np.mean(emb_matrix, axis=0)

    # normalize mean embedding (important for cosine similarity)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    return {
        "success": True,
        "faces_processed": len(raw_embeddings),

        "raw_embeddings": [
            emb.tolist() for emb in raw_embeddings
        ],

        "mean_embedding": mean_embedding.tolist(),

        "meta": faces_meta
    }