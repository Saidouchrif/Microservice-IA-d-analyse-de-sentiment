from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from BaseModel import Item
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from Model import pipe
app=FastAPI()

templings = Jinja2Templates(directory="../Front-end/src")
@app.get('/')
def home():
    return templings.TemplateResponse("Home.html", {"request": {"welcome":"welcome to my app"}})

@app.get('/model')
def get_model():
    return templings.TemplateResponse("ModelAi.html", {"request": {"welcome":"welcome to my app"}})

@app.post('/model')
def analyze_sentiment(request: Request, text: str = Form(...)):
    try:
        # Vérifier que le texte n'est pas vide
        if not text.strip():
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
        
        # Appel du modèle
        result = pipe(text)[0]

        # Préparer le JSON avec renommage label -> sentiment
        result_json = {
            "text": text,
            "result": {
                "sentiment": result.get("label", "UNKNOWN"),
                "score": result.get("score", 0.0)
            }
        }

        return templings.TemplateResponse("ModelAi.html", {
            "request": request,
            "result": result_json
        })

    except HTTPException as e:
        # Erreur connue (ex. texte vide)
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    except Exception as e:
        # Autres erreurs (modèle, pipeline, etc.)
        return JSONResponse(status_code=500, content={"error": f"Une erreur est survenue : {str(e)}"})

