import sys
import argparse
import statistics
import time
from pathlib import Path
import importlib.util
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "Back-end"


def load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def get_app_and_pipe():
    """Dynamically load FastAPI app and the model pipeline the same way tests do."""
    model_path = BACKEND_DIR / "Model.py"
    main_path = BACKEND_DIR / "main.py"

    # Ensure relative imports inside main.py work
    model_module = load_module("Back_end.Model", model_path)
    main_module = load_module("Back_end.main", main_path)
    return main_module.app, model_module.pipe, main_module


def format_stats(latencies):
    if not latencies:
        return {
            "count": 0,
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
        "count": count,
        "avg_ms": (total_time / count) * 1000.0,
        "p50_ms": p50 * 1000.0,
        "p95_ms": p95 * 1000.0,
        "min_ms": min(latencies) * 1000.0,
        "max_ms": max(latencies) * 1000.0,
        "throughput_rps": count / total_time if total_time > 0 else 0.0,
    }


def bench_pipe(pipe, samples, iterations):
    latencies = []
    # Warmup
    for text in samples:
        _ = pipe(text)

    for i in range(iterations):
        text = samples[i % len(samples)]
        start = time.perf_counter()
        _ = pipe(text)
        end = time.perf_counter()
        latencies.append(end - start)
    return latencies


def bench_endpoint(app, samples, iterations):
    from starlette.testclient import TestClient

    client = TestClient(app)
    latencies = []

    # Warmup
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


def main():
    parser = argparse.ArgumentParser(description="Mini-benchmark for sentiment analysis service")
    parser.add_argument("--iterations", type=int, default=20, help="Number of measured iterations per target")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=[
            "I love this product, it works amazingly well!",
            "This is terrible, I would not recommend it to anyone.",
            "It is okay, not great but not bad either.",
        ],
        help="Sample texts to use during the benchmark",
    )
    args = parser.parse_args()

    app, pipe, _ = get_app_and_pipe()

    print("Running mini-benchmark...")

    pipe_latencies = bench_pipe(pipe, args.samples, args.iterations)
    ep_latencies = bench_endpoint(app, args.samples, args.iterations)

    pipe_stats = format_stats(pipe_latencies)
    ep_stats = format_stats(ep_latencies)

    def pretty(stats):
        return (
            f"n={stats['count']} | avg={stats['avg_ms']:.1f} ms | p50={stats['p50_ms']:.1f} ms | "
            f"p95={stats['p95_ms']:.1f} ms | min={stats['min_ms']:.1f} ms | max={stats['max_ms']:.1f} ms | "
            f"throughput={stats['throughput_rps']:.2f} rps"
        )

    print("\nResults:")
    print("- Model pipeline (direct):  " + pretty(pipe_stats))
    print("- HTTP endpoint (/model):   " + pretty(ep_stats))


if __name__ == "__main__":
    main()


