from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from .Model import pipe

app = FastAPI()

# Dossier des templates
from pathlib import Path
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent  # remonte de Back-end/ à la racine
TEMPLATE_DIR = BASE_DIR / "Front-end" / "src"
templings = Jinja2Templates(directory=str(TEMPLATE_DIR))

@app.get("/")
def home(request: Request):
    return templings.TemplateResponse("Home.html", {"request": request})

@app.get("/model")
def get_model(request: Request):
    return templings.TemplateResponse("ModelAI.html", {"request": request})

@app.post("/model")
def analyze_sentiment(request: Request, text: str = Form(...)):
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
        
        # Appel du modèle
        result = pipe(text)[0]

        result_json = {
            "text": text,
            "result": {
                "sentiment": result.get("label", "UNKNOWN"),
                "score": result.get("score", 0.0)
            }
        }

        return templings.TemplateResponse("ModelAI.html", {
            "request": request,
            "result": result_json
        })

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Une erreur est survenue : {str(e)}"})
