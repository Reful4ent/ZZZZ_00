"""Split embeddings with a weights-hash cache (M-D14)."""
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expcore.io import atomic_write_bytes
from metric.errors import RunStop
from metric.pipeline import eval_arrays

EVAL_BATCH = 32


@dataclass
class EmbeddingSet:
    embeddings: np.ndarray
    image_ids: np.ndarray
    weights_hash: str
    split: str
    degradation_id: str
    method_id: str
    manifest_hash: str = ""
    preprocess_key: str = ""


def save_embeddings(path: Path, es: EmbeddingSet) -> None:
    buf = io.BytesIO()
    np.savez(buf, embeddings=es.embeddings.astype(np.float32), image_ids=np.asarray(es.image_ids, dtype=str),
             weights_hash=es.weights_hash, split=es.split, degradation_id=es.degradation_id,
             method_id=es.method_id, d=es.embeddings.shape[1], manifest_hash=es.manifest_hash,
             preprocess_key=es.preprocess_key)
    atomic_write_bytes(path, buf.getvalue())


def load_embeddings(path: Path) -> EmbeddingSet:
    with np.load(path, allow_pickle=False) as z:
        return EmbeddingSet(z["embeddings"], z["image_ids"], str(z["weights_hash"]), str(z["split"]),
                            str(z["degradation_id"]), str(z["method_id"]), str(z["manifest_hash"]),
                            str(z["preprocess_key"]))


def embed_split(method, images, split, store, params, degradation_id, seed, cache_dir, manifest_hash):
    wh = method.weights_hash()
    inp = params["input"]
    pk = json.dumps({"resize_short": int(inp["resize_short"]), "crop": int(inp["crop"]),
                     "noise_seed": seed if degradation_id.startswith("noise") else None}, sort_keys=True)
    path = Path(cache_dir) / f"{split}-{degradation_id}-{wh[:12]}.npz"
    if path.exists():
        try:
            es = load_embeddings(path)
            if (es.weights_hash, es.manifest_hash, es.split, es.degradation_id) == \
                    (wh, manifest_hash, split, degradation_id) and es.preprocess_key == pk:
                return es
        except Exception:  # corrupt/foreign file: recompute and overwrite
            pass
    ids = sorted(images.loc[images["split"] == split, "image_id"])
    chunks = [np.asarray(method.embed_for_search(x), dtype=np.float32)
              for _, x in eval_arrays(ids, store, inp["resize_short"], inp["crop"], EVAL_BATCH,
                                      degradation_id, seed)]
    z = np.concatenate(chunks) if chunks else np.zeros((0, 0), np.float32)
    if not np.isfinite(z).all():
        raise RunStop("INVALID_EMBEDDING_NORM", {"split": split, "degradation_id": degradation_id})
    es = EmbeddingSet(z, np.array(ids, dtype=str), wh, split, degradation_id, method.method_id, manifest_hash, pk)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    save_embeddings(path, es)
    return es
