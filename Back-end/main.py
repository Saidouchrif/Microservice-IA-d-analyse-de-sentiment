from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from BaseModel import Item
from fastapi.responses import JSONResponse
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
    result = pipe(text)[0]
    # renommage du label en sentiment
    result_json = {
        "text": text,
        "result": {
            "sentiment": result["label"],
            "score": result["score"]
        }
    }
    return templings.TemplateResponse("ModelAi.html", {
        "request": request,
        "result": result_json
    })

