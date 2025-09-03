
# Auto-generated integration orchestrator
# This file is meant to glue your disease classifier and your RAG system together.
# Edit as needed.

from typing import List, Dict, Tuple, Any

# Import the converted notebook modules
import importlib.util, types, json

def _load_module(path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

CLF_PATH = "disease_classifier_module.py"
RAG_PATH = "rag_module.py"

clf_mod = _load_module(CLF_PATH)
rag_mod = _load_module(RAG_PATH)

# Heuristics for classifier: pick a function that looks like a predictor
PREDICT_FN_CANDIDATES = []
def _get_classifier_predict():
    for name in PREDICT_FN_CANDIDATES:
        fn = getattr(clf_mod, name, None)
        if callable(fn):
            return fn
    # common fallbacks
    for name in dir(clf_mod):
        if name.lower().startswith("predict"):
            fn = getattr(clf_mod, name)
            if callable(fn):
                return fn
    raise RuntimeError("Could not find a classifier predict function. "
                       "Please expose a callable like `predict_diseases(symptoms_text) -> List[Tuple[str, float]]`.")

# Heuristics for RAG query: prefer a function that returns treatment/info for a disease
RAG_FN_CANDIDATES = []
def _get_rag_query():
    for name in RAG_FN_CANDIDATES:
        fn = getattr(rag_mod, name, None)
        if callable(fn):
            return fn

    # Try common names
    for name in ["retrieve_treatments", "get_treatments", "query_treatments", "rag_query", "ask_rag"]:
        fn = getattr(rag_mod, name, None)
        if callable(fn):
            return fn

    # Try chain-like objects with .run / .invoke
    for name in dir(rag_mod):
        obj = getattr(rag_mod, name)
        if hasattr(obj, "run"):
            return lambda disease: obj.run(f"What are the treatments and medical details for {disease}?")
        if hasattr(obj, "invoke"):
            return lambda disease: obj.invoke(f"What are the treatments and medical details for {disease}?")

    raise RuntimeError("Could not find a RAG query function. "
                       "Please expose `retrieve_treatments(disease_name: str) -> str` or a chain with .run/.invoke.")

def get_disease_and_treatments(symptoms: str, top_k: int = 5) -> Dict[str, Any]:
    """End-to-end pipeline.
    1) Uses the classifier to get candidate diseases and confidences.
    2) Uses RAG to fetch treatment/info for the top candidates.
    Returns a dict: {
        'input': symptoms,
        'results': [{'disease': str, 'confidence': float, 'info': str}]
    }
    """
    predict = _get_classifier_predict()
    rag_query = _get_rag_query()

    preds = predict(symptoms)  # expected: List[Tuple[str, float]]
    # Defensive normalization
    norm = []
    if isinstance(preds, dict):
        # if dict like {'Disease': prob, ...}
        norm = sorted([(k, float(v)) for k, v in preds.items()], key=lambda x: x[1], reverse=True)
    elif isinstance(preds, list):
        for item in preds:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                disease = str(item[0])
                conf = float(item[1]) if len(item) > 1 else 0.0
                norm.append((disease, conf))
            else:
                disease = str(item)
                norm.append((disease, 0.0))
        norm.sort(key=lambda x: x[1], reverse=True)
    else:
        # unknown format: make a single entry
        norm = [(str(preds), 0.0)]

    norm = norm[:top_k]

    results = []
    for disease, conf in norm:
        try:
            info = rag_query(disease)
        except Exception as e:
            info = f"[RAG query failed: {type(e).__name__}: {e}]"
        results.append({'disease': disease, 'confidence': float(conf), 'info': str(info)})

    return {
        'input': symptoms,
        'results': results
    }
