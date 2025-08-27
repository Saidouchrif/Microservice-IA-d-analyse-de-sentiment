import sys
from pathlib import Path
import importlib.util
from types import ModuleType
import pytest
from starlette.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "Back-end"


def load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def get_app_module():
    # Précharger Model sous un alias compatible pour l'import relatif
    model_path = BACKEND_DIR / "Model.py"
    load_module("Back_end.Model", model_path)
    # Charger main avec le même nom de paquet
    main_path = BACKEND_DIR / "main.py"
    main_module = load_module("Back_end.main", main_path)
    return main_module


@pytest.fixture(scope="module")
def app_module():
    return get_app_module()


@pytest.fixture()
def client(app_module):
    return TestClient(app_module.app)


def test_analyze_sentiment_success(client, app_module, monkeypatch):
    def fake_pipe(text):
        return [{"label": "POSITIVE", "score": 0.9876}]

    monkeypatch.setattr(app_module, "pipe", fake_pipe, raising=True)

    response = client.post("/model", data={"text": "J'adore ce produit!"})
    assert response.status_code == 200
    assert getattr(response, "template").name == "ModelAI.html"
    context = getattr(response, "context")
    assert "result" in context
    result = context["result"]["result"]
    assert result["text"] == "J'adore ce produit!"
    assert result["sentiment"] == "POSITIVE"
    assert isinstance(result["score"], float)


def test_analyze_sentiment_empty_text_error(client):
    response = client.post("/model", data={"text": "   "})
    assert response.status_code == 200
    assert getattr(response, "template").name == "ModelAI.html"
    context = getattr(response, "context")
    assert context.get("error") == "Le texte ne peut pas être vide."


def test_analyze_sentiment_unexpected_exception(client, app_module, monkeypatch):
    def exploding_pipe(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "pipe", exploding_pipe, raising=True)

    response = client.post("/model", data={"text": "Quelque chose"})
    assert response.status_code == 200
    assert getattr(response, "template").name == "ModelAI.html"
    context = getattr(response, "context")
    assert "Une erreur est survenue" in context.get("error", "")
