
# Integrated Disease Classifier + RAG

This package stitches your **disease classifier** (from `disease_classifier.ipynb`) and your **RAG system** (from `chatbot.ipynb`) into a single pipeline and a small FastAPI service.

## Files generated
- `rag_module.py` — pythonized from your RAG notebook
- `disease_classifier_module.py` — pythonized from your classifier notebook
- `integrated_pipeline.py` — orchestrator glue code
- `app_fastapi.py` — FastAPI app exposing `/diagnose`

## Expected interfaces (heuristics & how to adapt)
- **Classifier:** expose a function like:

```python
def predict_diseases(symptoms_text: str) -> list[tuple[str, float]]:
    return [("Influenza", 0.81), ("Common cold", 0.62)]
```

If you already have a function named `predict_*`, we'll try to use it automatically. Dict outputs like `{"Disease": prob}` are also accepted.

- **RAG:** expose either
  - `retrieve_treatments(disease_name: str) -> str`  (preferred), or
  - any function whose name includes `retrieve|query|rag|treat`, or
  - a chain-like object with `.run()` or `.invoke()` that can accept a question.

We'll call it with `What are the treatments and medical details for {disease}?` if using a generic chain.

## How to run
1. Install deps if needed (FastAPI, uvicorn, plus anything your notebooks require):
```
pip install fastapi uvicorn langchain faiss-cpu transformers scikit-learn torch
```
2. Run the API:
```
python /mnt/data/app_fastapi.py
```
3. Example request:
```
POST http://localhost:8000/diagnose
{
  "symptoms": "fever, cough, sore throat",
  "top_k": 3
}
```

## Local usage as a library
```python
from integrated_pipeline import get_disease_and_treatments
print(get_disease_and_treatments("fever, cough, sore throat"))
```

## Notes
- If the orchestrator can't find your functions, edit `integrated_pipeline.py` to point directly to the right callables.
- Make sure your notebooks save/load any ML weights or vector stores they require.
