## Microservice IA d'analyse de sentiment

Application FastAPI avec interface HTML minimaliste pour analyser le sentiment d'un texte via un modèle Transformers, accompagnée d'un mini-benchmark, de tests, d'un exemple Gradio et d'un Dockerfile.

### Contenu du dépôt
```
Microservice-IA-d-analyse-de-sentiment/
  Back-end/                 # Application FastAPI (routes + chargement du modèle)
  Front-end/                # Templates HTML (Jinja2)
  Gradio/                   # Démonstration Gradio
  Docker/                   # Variante packagée + Dockerfile additionnel
  Perf_benchmark.py         # Mini-benchmark des performances
  Test_app.py               # Tests d'intégration FastAPI
  requirements.txt          # Dépendances principales
  Dockerfile                # Dockerfile à la racine
  README.md                 # Ce fichier
```

### Prérequis
- Python 3.9+ (recommandé 3.10)
- pip (ou pipx)
- Windows PowerShell, macOS Terminal ou Linux shell

### Installation (locale)
1) Créer un environnement virtuel et installer les dépendances
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell (Windows)
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

2) Premier lancement (le modèle sera téléchargé au premier appel)
```bash
uvicorn Back-end.main:app --reload
```

3) Ouvrir dans le navigateur
- Page d'accueil: `http://127.0.0.1:7860/`
- Page modèle: `http://127.0.0.1:7860/model`

### Endpoints (résumé)
- `GET /` → rend `Home.html`
- `GET /model` → rend `ModelAI.html`
- `POST /model` (Form) → champ `text` requis, exécute le pipeline et renvoie la page avec le résultat dans le contexte Jinja

### Tests
Exécuter la suite de tests (avec monkeypatch du pipeline) :
```bash
pytest -q
```

### Mini-benchmark (latence et débit)
Le script mesure la latence du pipeline directement et de l'endpoint `/model` via un client de test.

Commandes:
```bash
# exécution par défaut
python Perf_benchmark.py

# options personnalisées
python Perf_benchmark.py --iterations 30 --samples "Great job!" "Not good." "Meh, it's fine."
```
Le résultat affiche: moyenne, p50, p95, min, max et throughput (rps) pour:
- appel direct `pipe(text)`
- appel HTTP `POST /model`

Astuce: le premier run peut être plus lent (téléchargement du modèle + warmup).

### Démo Gradio
Un exemple Gradio indépendant est fourni.
```bash
cd Gradio/Sentiment-Gradio
pip install -r requirements.txt
python app.py
```
L'interface s'ouvre sur `http://127.0.0.1:7860` (par défaut).

### Docker (racine)
Construire l'image et lancer un conteneur exposant l'API:
```bash
docker build -t fastapi-sentiment .
docker run --rm -p 7860:7860 fastapi-sentiment
```
Accès: `http://127.0.0.1:7860/`

Note: Un dossier `Docker/` contient aussi une variante packagée; privilégiez le Dockerfile racine sauf besoin spécifique.

### Structure du code (points clés)
- `Back-end/main.py` :
  - charge les templates depuis `Front-end/src`
  - routes `/`, `/model` (GET/POST)
- `Back-end/Model.py` :
  - crée `pipe = pipeline("text-classification", model="distilbert/...-sst-2-english")`

### Problèmes fréquents
- Téléchargement du modèle: nécessite Internet lors du premier run.
- Mémoire GPU/CPU: le pipeline tourne sur CPU par défaut; assurez-vous d'avoir suffisamment de RAM.
- Windows PowerShell: si `python` n'est pas reconnu, essayez `py` à la place.

### Licence
Libre d'utilisation pour usage pédagogique et expérimentation. Adaptez selon vos besoins.
