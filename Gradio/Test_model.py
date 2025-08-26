import gradio as gr
from transformers import pipeline

# Charger ton modèle HuggingFace
pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Fonction d'analyse
def analyze_sentiment(text):
    if not text.strip():
        return {"error": "Le texte ne peut pas être vide."}
    
    result = pipe(text)[0]
    return {
        "text": text,
        "sentiment": result.get("label", "UNKNOWN"),
        "score": round(result.get("score", 0.0), 3)
    }

# Interface Gradio
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Tapez votre texte ici..."),
    outputs="json",
    title="🧠 Sentiment Analysis Model",
    description="Entrez un texte et obtenez le sentiment (positif/négatif)."
)

if __name__ == "__main__":
    # Pour Windows local -> 127.0.0.1
    # Pour Docker/Linux -> 0.0.0.0
    demo.launch(server_name="127.0.0.1", server_port=7890)
