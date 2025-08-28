import sys
import time
from pathlib import Path
import importlib.util
from types import ModuleType
from typing import List, Dict, Any

import gradio as gr
from transformers import pipeline as hf_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "Back-end"


def load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def get_app_and_pipe():
    model_path = BACKEND_DIR / "Model.py"
    main_path = BACKEND_DIR / "main.py"
    model_module = load_module("Back_end.Model", model_path)
    main_module = load_module("Back_end.main", main_path)
    return main_module.app, model_module.pipe


def get_two_pipes():
    """Return two pipelines for comparison: distilbert (pipe1) and roberta-large (pipe2)."""
    pipe1 = hf_pipeline(
        "text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    pipe2 = hf_pipeline(
        "text-classification",
        model="siebert/sentiment-roberta-large-english",
    )
    return pipe1, pipe2


def format_stats(latencies: List[float]) -> Dict[str, Any]:
    if not latencies:
        return {
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "throughput_rps": 0.0,
        }

    total_time = sum(latencies)
    count = len(latencies)
    sorted_lats = sorted(latencies)
    p50 = sorted_lats[int(0.50 * (count - 1))]
    p95 = sorted_lats[int(0.95 * (count - 1))]
    return {
        "avg_ms": (total_time / count) * 1000.0,
        "p50_ms": p50 * 1000.0,
        "p95_ms": p95 * 1000.0,
        "min_ms": min(latencies) * 1000.0,
        "max_ms": max(latencies) * 1000.0,
        "throughput_rps": count / total_time if total_time > 0 else 0.0,
    }


def bench_pipe(pipe, samples: List[str], iterations: int) -> List[float]:
    latencies: List[float] = []
    for text in samples:
        _ = pipe(text)
    for i in range(iterations):
        text = samples[i % len(samples)]
        start = time.perf_counter()
        _ = pipe(text)
        end = time.perf_counter()
        latencies.append(end - start)
    return latencies


def bench_endpoint(app, samples: List[str], iterations: int) -> List[float]:
    from starlette.testclient import TestClient

    client = TestClient(app)
    latencies: List[float] = []
    for text in samples:
        client.post("/model", data={"text": text})
    for i in range(iterations):
        text = samples[i % len(samples)]
        start = time.perf_counter()
        resp = client.post("/model", data={"text": text})
        _ = resp.text
        end = time.perf_counter()
        latencies.append(end - start)
    return latencies


def run_benchmark(samples_text: str, iterations: int):
    samples = [s.strip() for s in samples_text.splitlines() if s.strip()]
    if not samples:
        raise gr.Error("Veuillez fournir au moins une phrase d'exemple (une par ligne).")

    app, pipe_fastapi = get_app_and_pipe()
    pipe1, pipe2 = get_two_pipes()

    pipe1_lat = bench_pipe(pipe1, samples, iterations)
    pipe2_lat = bench_pipe(pipe2, samples, iterations)
    ep_lat = bench_endpoint(app, samples, iterations)

    pipe1_stats = format_stats(pipe1_lat)
    pipe2_stats = format_stats(pipe2_lat)
    ep_stats = format_stats(ep_lat)

    headers = [
        "cible",
        "avg_ms",
        "p50_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "throughput_rps",
    ]

    rows = [
        [
            "pipe1 DistilBERT (direct)",
            round(pipe1_stats["avg_ms"], 1),
            round(pipe1_stats["p50_ms"], 1),
            round(pipe1_stats["p95_ms"], 1),
            round(pipe1_stats["min_ms"], 1),
            round(pipe1_stats["max_ms"], 1),
            round(pipe1_stats["throughput_rps"], 2),
        ],
        [
            "pipe2 RoBERTa-large (direct)",
            round(pipe2_stats["avg_ms"], 1),
            round(pipe2_stats["p50_ms"], 1),
            round(pipe2_stats["p95_ms"], 1),
            round(pipe2_stats["min_ms"], 1),
            round(pipe2_stats["max_ms"], 1),
            round(pipe2_stats["throughput_rps"], 2),
        ],
        [
            "endpoint (/model)",
            round(ep_stats["avg_ms"], 1),
            round(ep_stats["p50_ms"], 1),
            round(ep_stats["p95_ms"], 1),
            round(ep_stats["min_ms"], 1),
            round(ep_stats["max_ms"], 1),
            round(ep_stats["throughput_rps"], 2),
        ],
    ]

    # Example predictions for first sample to visualize qualitative difference
    example = samples[0]
    pred1 = pipe1(example)[0]
    pred2 = pipe2(example)[0]
    preds = [
        ["exemple", "pipe1 label", "pipe1 score", "pipe2 label", "pipe2 score"],
        [
            example,
            pred1.get("label", ""),
            round(float(pred1.get("score", 0.0)), 4),
            pred2.get("label", ""),
            round(float(pred2.get("score", 0.0)), 4),
        ],
    ]

    return rows, preds


with gr.Blocks(title="Benchmark Sentiment (6 critères)") as demo:
    gr.Markdown("**Benchmark de performance sur 6 critères : avg, p50, p95, min, max, throughput**\n\nComparaison: pipe1 (DistilBERT) vs pipe2 (RoBERTa-large), et endpoint /model.")
    with gr.Row():
        samples = gr.Textbox(
            label="Phrases d'exemple (une par ligne)",
            value=(
                "I love this product, it works amazingly well!\n"
                "This is terrible, I would not recommend it to anyone.\n"
                "It is okay, not great but not bad either."
            ),
            lines=6,
        )
        iterations = gr.Slider(5, 50, value=20, step=1, label="Itérations mesurées par cible")
    run_btn = gr.Button("Lancer le benchmark")
    table = gr.Dataframe(headers=["cible", "avg_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "throughput_rps"], row_count=3)
    preds = gr.Dataframe(headers=["exemple", "pipe1 label", "pipe1 score", "pipe2 label", "pipe2 score"], row_count=2)

    run_btn.click(run_benchmark, inputs=[samples, iterations], outputs=[table, preds])


if __name__ == "__main__":
    demo.launch()


