"""
identification/registry.py
---------------------------
Stores ALL enrollment embeddings separately (not averaged).
Scores a query against every stored embedding and takes the max.
This makes identification robust across server restarts because
one good enrollment session is enough to anchor the identity.
"""

import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from identification.titanet import get_embedding_windowed as get_embedding

ENROLLED_DIR = os.path.join(os.path.dirname(__file__), "..", "enrolled_speakers")

def _get_thresholds():
    threshold = float(os.environ.get("TITANET_THRESHOLD", "0.55"))
    margin = float(os.environ.get("TITANET_MARGIN", "0.05"))
    return threshold, margin


def _safe_name(name: str) -> str:
    clean = re.sub(r"[^\w\- ]", "", name or "").strip()
    if not clean:
        raise ValueError(f"Invalid speaker name: {name!r}")
    return clean


def enroll_speaker(name: str, wav_path: str) -> dict:
    """
    Save each enrollment as a separate .npy file:
        enrolled_speakers/<name>_0.npy
        enrolled_speakers/<name>_1.npy
        ...
    Multiple files = more robust matching across sessions.
    """
    name = _safe_name(name)
    os.makedirs(ENROLLED_DIR, exist_ok=True)

    new_emb = get_embedding(wav_path)

    # Find next available index for this speaker
    existing = _get_speaker_files(name)
    idx      = len(existing)
    save_path = os.path.join(ENROLLED_DIR, f"{name}_{idx}.npy")

    np.save(save_path, new_emb)
    total = idx + 1
    print(f"[registry] Enrolled '{name}' sample {total} → {save_path}")

    MIN_RECOMMENDED_SAMPLES = 3

    return {
        "name":            name,
        "status":          "enrolled",
        "sample_count":    total,
        "total_enrolled":  len(list_enrolled()),
        "needs_more_samples": total < MIN_RECOMMENDED_SAMPLES,
    }


def identify_cluster_embedding(cluster_emb: np.ndarray, fallback_label: str = "Unknown") -> dict:
    all_embeddings = _load_all_enrolled()
    if not all_embeddings:
        return {"name": fallback_label, "score": 0.0, "reason": "no enrolled speakers"}

    query_emb = cluster_emb.reshape(1, -1)
    scores = {}
    for name, emb_list in all_embeddings.items():
        centroid = np.mean(emb_list, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        scores[name] = float(cosine_similarity(query_emb, centroid.reshape(1, -1))[0][0])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0

    print(f"[registry] cluster scores: {scores} | best={best_name} ({best_score:.3f})")

    threshold, margin = _get_thresholds()

    if best_score < threshold:
        return {"name": fallback_label, "score": round(best_score, 3), "reason": f"below threshold ({best_score:.3f} < {threshold})"}

    if best_score - second_score < margin:
        return {"name": fallback_label, "score": round(best_score, 3), "reason": f"ambiguous — top 2 too close ({best_score:.3f} vs {second_score:.3f}, diff < {margin})"}

    return {"name": best_name, "score": round(best_score, 3)}


def identify_speaker(wav_path: str, fallback_label: str = "Unknown") -> dict:
    try:
        query_emb = get_embedding(wav_path)
    except Exception as e:
        return {"name": fallback_label, "score": 0.0, "reason": str(e)}

    return identify_cluster_embedding(query_emb, fallback_label=fallback_label)

def list_enrolled() -> list:
    """Return unique speaker names (deduplicated across multiple sample files)."""
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    names = set()
    for f in os.listdir(ENROLLED_DIR):
        if f.endswith(".npy"):
            # Strip trailing _0, _1, _2 suffix to get base name
            base = re.sub(r"_\d+$", "", f[:-4])
            names.add(base)
    return sorted(names)


def delete_speaker(name: str) -> dict:
    """Delete all enrollment files for a speaker."""
    name  = _safe_name(name)
    files = _get_speaker_files(name)
    if not files:
        return {"name": name, "status": "not_found"}
    for f in files:
        os.remove(f)
    print(f"[registry] Deleted {len(files)} enrollment(s) for '{name}'")
    return {"name": name, "status": "deleted", "files_removed": len(files)}


def _get_speaker_files(name: str) -> list:
    """Return all .npy file paths for a given speaker name."""
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(name)}_\d+\.npy$")
    return sorted(
        os.path.join(ENROLLED_DIR, f)
        for f in os.listdir(ENROLLED_DIR)
        if pattern.match(f)
    )


def _load_all_enrolled() -> dict:
    """
    Returns { speaker_name: [emb_array_0, emb_array_1, ...] }
    Loads all individual enrollment files per speaker.
    """
    os.makedirs(ENROLLED_DIR, exist_ok=True)
    result = {}

    for f in os.listdir(ENROLLED_DIR):
        if not f.endswith(".npy"):
            continue
        base = re.sub(r"_\d+$", "", f[:-4])   # "joy_2" → "joy"
        path = os.path.join(ENROLLED_DIR, f)
        emb  = np.load(path).astype(np.float64)

        # Re-normalise on load in case file was saved before norm fix
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm

        if base not in result:
            result[base] = []
        result[base].append(emb)

    return result