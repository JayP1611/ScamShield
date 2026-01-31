import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
from joblib import dump, load
from scipy.sparse import hstack
import tensorflow as tf
from pathlib import Path

from ml.features import extract_handcrafted_features

# ---------------------------------- PATHS -------------------------------------------------------
# LOGREG_DIR = "../artifacts/log_reg"
# NN_DIR = "../artifacts/neural_net"
#
# VEC_PATH = os.path.join(LOGREG_DIR, "vec.joblib")
# LOGREG_PATH = os.path.join(LOGREG_DIR, "mod_logreg.joblib")
# FEATURE_CFG_PATH = os.path.join(LOGREG_DIR, "feature_cfg.joblib")
#
# NN_MODEL_PATH = os.path.join(NN_DIR, "scamshield_nn_tf.keras")

BASE_DIR = Path(__file__).resolve().parents[1]  # project root

LOGREG_DIR = BASE_DIR / "artifacts" / "log_reg"
NN_DIR = BASE_DIR / "artifacts" / "neural_net"

VEC_PATH = LOGREG_DIR / "vec.joblib"
LOGREG_PATH = LOGREG_DIR / "mod_logreg.joblib"
FEATURE_CFG_PATH = LOGREG_DIR / "feature_config.json"

NN_MODEL_PATH = NN_DIR / "scamshield_nn_tf.keras"

# Clustering
CLUSTER_DIR = BASE_DIR / "artifacts" / "clustering"
KMEANS_PATH = CLUSTER_DIR / "kmeans_tfidf.joblib"
CLUSTER_VEC_PATH = CLUSTER_DIR / "cluster_vectorizer.joblib"
CLUSTER_TERMS_PATH = CLUSTER_DIR / "cluster_terms.json"


# ------------------------------------ HELPERS -----------------------------------------------------
def _load_feature_config():
    if not os.path.exists(FEATURE_CFG_PATH):
        return {
            "handcrafted_features": ["num_urls", "num_digits", "msg_len", "num_exclaim", "has_currency", "has_urgent"]
        }
    with open(FEATURE_CFG_PATH, 'r') as f:
        cfg = json.load(f)
    return cfg


def _verdict_from_prob(p: float, threshold: float = 0.5) -> str:
    return "Scam" if p >= threshold else "Ham"


def _scipy_to_tf_sparse(X):
    """
    Convert SciPy sparse matrix -> tf.sparse.SparseTensor
    so Keras can accept sparse TF-IDF input.
    """
    X = X.tocoo()
    indices = np.stack([X.row, X.col], axis=1).astype(np.int64)
    values = X.data.astype(np.float32)
    shape = np.array(X.shape, dtype=np.int64)
    st = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
    return tf.sparse.reorder(st)


def _top_logreg_terms(vec, logreg_model, text: str, top_k: int = 8):
    """
    Explainability for LogReg:
    Find the top positive contributing TF-IDF terms for THIS message.
    Works because logistic regression is linear: contribution ~ weight * tfidf_value.
    """
    X = vec.transform([text])  # (1, vocab)
    feature_names = vec.get_feature_names_out()
    weights = logreg_model.coef_.reshape(-1)  # weights for class 1

    # Get non-zero indices in this message's tfidf vector
    row = X.tocoo()
    contribs = []
    for idx, val in zip(row.col, row.data):
        contribs.append((feature_names[idx], float(val * weights[idx])))

    # Sort by contribution descending (more positive => more scammy)
    contribs.sort(key=lambda x: x[1], reverse=True)
    # Keep only positive contributions
    contribs = [(t, c) for (t, c) in contribs if c > 0]

    return contribs[:top_k]


# -------- Main API --------
class ScamShieldPredictor:
    def __init__(self):
        if not VEC_PATH.exists():
            raise FileNotFoundError(f"Vectorizer not found: {VEC_PATH}")

        if not LOGREG_PATH.exists():
            raise FileNotFoundError(f"LogReg model not found: {LOGREG_PATH}")

        self.vec = load(str(VEC_PATH))
        self.logreg = load(str(LOGREG_PATH))
        self.feature_cfg = _load_feature_config()

        # Lazy-load NN (loads only when needed)
        self._nn_model = None

        # Clustering
        self._cluster_vec = None
        self._kmeans = None
        self._cluster_terms = None

    def _get_nn(self):
        if self._nn_model is None:
            if not NN_MODEL_PATH.exists():
                raise FileNotFoundError(f"NN model not found: {NN_MODEL_PATH}")
            self._nn_model = tf.keras.models.load_model(NN_MODEL_PATH)
        return self._nn_model

    def _build_features(self, text: str):
        # TF-IDF
        X_tfidf = self.vec.transform([text])

        # Handcrafted
        X_hand = extract_handcrafted_features([text])  # shape (1, 6)

        # Combined (for LogReg)
        X_full = hstack([X_tfidf, X_hand])

        return X_tfidf, X_hand, X_full

    def _load_clustering(self):
        if self._cluster_vec is None:
            if not CLUSTER_VEC_PATH.exists():
                raise FileNotFoundError(f"Cluster vector model not found: {CLUSTER_VEC_PATH}")
            self._cluster_vec = load(CLUSTER_VEC_PATH)

        if self._kmeans is None:
            if not KMEANS_PATH.exists():
                raise FileNotFoundError(f"Kmeans model not found: {KMEANS_PATH}")
            self._kmeans = load(KMEANS_PATH)

        if self._cluster_terms is None:
            if not CLUSTER_TERMS_PATH.exists():
                raise FileNotFoundError(f"Cluster terms json not found: {CLUSTER_TERMS_PATH}")
            with open(CLUSTER_TERMS_PATH, "r", encoding="utf-8") as f:
                self._cluster_terms = json.load(f)

        return self._cluster_vec, self._kmeans, self._cluster_terms

    def _get_cluster_insight(self, text: str, top_n: int = 6):
        vec, kmeans, terms = self._load_clustering()
        x = vec.transform([text])
        cid = int(kmeans.predict(x)[0])
        top_terms = terms.get(str(cid), [])[:top_n]
        return cid, top_terms

    def predict(self, text: str, model: str = "logreg", threshold: float = 0.5, return_embedding: bool = True) -> dict:
        """
        model: 'logreg' or 'nn'
        return_embedding: only relevant for NN
        """
        text = str(text).strip()
        if len(text) == 0:
            return {
                "error": "Empty text. Please provide a message.",
                "model_used": model
            }

        X_tfidf, X_hand, X_full = self._build_features(text)

        result = {
            "model_used": model,
            "threshold": float(threshold),
            "handcrafted_features": {
                "num_urls": float(X_hand[0, 0]),
                "num_digits": float(X_hand[0, 1]),
                "msg_len": float(X_hand[0, 2]),
                "num_exclaim": float(X_hand[0, 3]),
                "has_currency": float(X_hand[0, 4]),
                "has_urgent": float(X_hand[0, 5]),
            }
        }

        # clustering insight
        try:
            cid, cterms = self._get_cluster_insight(text, top_n = 6)
            result["cluster"] = {"id": cid, "top_terms": cterms}
        except Exception as e:
            # incase the clustering file is mising, don't break the result until now and continue
            result["cluster"] = {"error": str(e)}

        # Logistic Regression
        if model == "logreg":
            p = float(self.logreg.predict_proba(X_full)[0, 1])
            label = int(p >= threshold)
            verdict = _verdict_from_prob(p, threshold)
            top_terms = _top_logreg_terms(self.vec, self.logreg, text, top_k=8)

            result.update({
                "probability": p,
                "risk_score": int(round(p * 100)),
                "label": label,
                "verdict": verdict,
                "explanations": {
                    "top_scam_terms": [{"term": t, "contribution": c} for (t, c) in top_terms]
                }
            })
            return result

        # Neural network
        if model == "nn":
            nn = self._get_nn()

            X_tfidf_tf = _scipy_to_tf_sparse(X_tfidf)
            preds = nn.predict({"tfidf": X_tfidf_tf, "handcrafted": X_hand}, verbose=0)

            # model outputs dict: {"prob":..., "embedding":...}
            p = float(np.array(preds["prob"]).reshape(-1)[0])
            label = int(p >= threshold)
            verdict = _verdict_from_prob(p, threshold)

            out = {
                "probability": p,
                "risk_score": int(round(p * 100)),
                "label": label,
                "verdict": verdict
            }

            if return_embedding:
                emb = np.array(preds["embedding"]).reshape(-1).astype(float).tolist()
                out["embedding"] = emb

            result.update(out)
            return result

        return {"error": f"Unknown model '{model}'. Use 'logreg' or 'nn'."}


# -------- Quick test runner --------
# if __name__ == "__main__":
#     predictor = ScamShieldPredictor()
#
#     msg = "URGENT! Your account is suspended. Verify now: http://bit.ly/abcd"
#     print("\nLOGREG:\n", predictor.predict(msg, model="logreg"))
#     print("\nNN:\n", predictor.predict(msg, model="nn", return_embedding=True))
