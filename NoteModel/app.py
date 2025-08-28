from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn


pipe2 = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")

app = FastAPI(title="Sentiment Test API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, Any]:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    try:
        result = pipe2(text)[0]
        return {
            "text": text,
            "label": result.get("label", "UNKNOWN"),
            "score": float(result.get("score", 0.0)),
        }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)