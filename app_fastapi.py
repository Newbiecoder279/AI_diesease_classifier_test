
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from integrated_pipeline import get_disease_and_treatments

app = FastAPI(title="Disease Diagnosis + RAG API", version="1.0.0")

class DiagnoseRequest(BaseModel):
    symptoms: str
    top_k: int = 5

@app.post("/diagnose")
def diagnose(req: DiagnoseRequest) -> Dict[str, Any]:
    return get_disease_and_treatments(req.symptoms, top_k=req.top_k)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
