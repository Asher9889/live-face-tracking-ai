from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
import cv2
import numpy as np
import threading
import time 
import requests

from app.recognition import embedding_store, unknown_embedding_store
from app.ai.insight_detector import InsightFaceEngine

app = FastAPI()
insight_engine = InsightFaceEngine()

lock = threading.Lock()

from pydantic import BaseModel, Field
from typing import List

EXPECTED_DIM = 512
class MergeRequest(BaseModel):
    embeddings: List[List[float]] = Field(..., min_items=2)
    counts: List[int] = Field(..., min_items=2)

class DuplicateCheckRequest(BaseModel):
    embedding: List[float]
    threshold: Optional[float] = 0.5

class DuplicateCheckResponse(BaseModel):
    isDuplicate: bool
    employeeId: Optional[str] = None
    similarity: Optional[float] = None


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def wait_for_api(timeout=10):
    start = time.time()

    while time.time() - start < timeout:
        try:
            res = requests.get("http://localhost:4001/health")
            if res.status_code == 200:
                return True
        except:
            pass

        time.sleep(0.5)

    return False


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


@app.post("/merge")
def merge_embeddings(data: MergeRequest):
    try:
        embeddings = np.array(data.embeddings, dtype=np.float32)
        counts = np.array(data.counts, dtype=np.float32)

        # 1. Basic validation
        if len(embeddings) < 2:
            return {"status": "error", "message": "Need at least 2 embeddings"}

        if len(embeddings) != len(counts):
            return {"status": "error", "message": "Embeddings and counts mismatch"}

        if np.any(counts <= 0):
            return {"status": "error", "message": "Counts must be positive"}

        if embeddings.shape[1] != EXPECTED_DIM:
            return {
                "status": "error",
                "message": f"Invalid embedding dimension. Expected {EXPECTED_DIM}, got {embeddings.shape[1]}"
            }

        # 2. Normalize embeddings
        normalized = []
        for e in embeddings:
            n = normalize(e)
            if n is None:
                return {"status": "error", "message": "Zero norm embedding detected"}
            normalized.append(n)

        embeddings = np.array(normalized)

        # 3. Similarity validation
        min_similarity = 1.0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                min_similarity = min(min_similarity, sim)

                if sim < 0.7:
                    return {
                        "status": "error",
                        "message": f"Embeddings too different (similarity={sim:.3f})"
                    }

        # 4. Weighted merge
        weighted_sum = np.sum(embeddings * counts[:, None], axis=0)
        merged = weighted_sum / np.sum(counts)

        merged = normalize(merged)
        if merged is None:
            return {"status": "error", "message": "Merged embedding invalid"}

        # 5. Confidence metrics
        sims = [cosine_similarity(merged, e) for e in embeddings]

        return {
            "status": "success",
            "mergedEmbedding": merged.tolist(),
            "totalCount": int(np.sum(counts)),
            "minSimilarity": float(min_similarity),
            "avgSimilarity": float(np.mean(sims))
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Merge failed: {str(e)}"
        }

@app.post("/recognition/check-duplicate")
def check_duplicate(data: DuplicateCheckRequest):
    try:
        embedding = data.embedding
        threshold = data.threshold or 0.5

        # Validate embedding
        if not embedding or len(embedding) == 0:
            return {
                "success": False,
                "error": "Empty embedding",
                "code": "INVALID_INPUT"
            }

        emb = np.array(embedding, dtype=np.float32)

        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return {
                "success": False,
                "error": "Invalid embedding (zero norm)",
                "code": "INVALID_EMBEDDING"
            }

        emb = emb / norm

        # Find match
        match = embedding_store.find_match(emb)

        if not match:
            return {
                "success": True,
                "isDuplicate": False,
                "similarity": None
            }

        similarity = float(match["similarity"])
        employee_id = match["employee_id"]

        # Apply threshold

        if similarity >= threshold:
            return {
                "success": True,
                "isDuplicate": True,
                "employeeId": employee_id,
                "similarity": similarity
            }

        return {
            "success": True,
            "isDuplicate": False,
            "similarity": similarity
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "code": "INTERNAL_ERROR"
        }

class PromoteRequest(BaseModel):
    unknownId: str
    employeeId: str
    employeeName: str
    embedding: list[float]

@app.post("/recognition/promote-unknown")
def promote_unknown(data: PromoteRequest):
    try:
        emb = np.array(data.embedding, dtype=np.float32)

        # Validate embedding
        if emb.shape[0] != 512:
            return {
                "success": False,
                "error": "Invalid embedding dimension"
            }

        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return {
                "success": False,
                "error": "Invalid embedding (zero norm)"
            }

        emb = emb / norm

        with lock:

            # Validate unknown exists

            if data.unknownId not in unknown_embedding_store.id_to_index:
                return {
                    "success": False,
                    "error": "Unknown ID not found in AI store"
                }

            # Prevent duplicate employee
            if data.employeeId in embedding_store.employee_ids:
                return {
                    "success": True,
                    "message": "Already promoted"
                }

            # REMOVE from unknown store
            idx = unknown_embedding_store.id_to_index[data.unknownId]

            unknown_embedding_store.embeddings = np.delete(
                unknown_embedding_store.embeddings, idx, axis=0
            )

            unknown_embedding_store.counts.pop(idx)
            unknown_embedding_store.unknown_ids.pop(idx)

            # rebuild index map
            unknown_embedding_store.id_to_index = {
                uid: i for i, uid in enumerate(unknown_embedding_store.unknown_ids)
            }

            # ADD to employee store
            embedding_store.employee_ids.append(data.employeeId)
            embedding_store.employee_names.append(data.employeeName)

            if (
                embedding_store.embeddings is None
                or embedding_store.embeddings.shape[0] == 0
            ):
                embedding_store.embeddings = emb.reshape(1, -1)
            else:
                embedding_store.embeddings = np.vstack([
                    embedding_store.embeddings,
                    emb
                ])

        print(f"[AI] Promoted unknown {data.unknownId} → {data.employeeId}")

        return {
            "success": True,
            "message": "Promoted successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }