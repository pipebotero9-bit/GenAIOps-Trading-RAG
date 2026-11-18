# app/resumen_informe.py
"""
Cifras para INFORME.md: correctness por tipo de pregunta, latencia y tokens
de cada corrida de evaluación (leídos de las trazas).

Uso:
    python app/resumen_informe.py
"""

import json
import os
import statistics
import sys
from collections import defaultdict

import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.tracking import EVAL_EXPERIMENT


def _si(v):
    return str(v).strip().lower() in ("yes", "true", "1")


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    tipos = {q["question"]: q.get("type", "?")
             for q in json.load(open("tests/eval_dataset.json", encoding="utf-8"))}
    runs = mlflow.search_runs(experiment_names=[EVAL_EXPERIMENT], order_by=["start_time ASC"])
    runs = runs[runs["params.prompt"].fillna("").str.contains("asistente_trading")]

    for _, r in runs.iterrows():
        traces = mlflow.search_traces(locations=[r["experiment_id"]], run_id=r["run_id"],
                                      return_type="list")
        if not traces:
            continue
        lat, tok_in, tok_out = [], [], []
        por_tipo = defaultdict(list)
        for t in traces:
            if t.info.execution_time_ms:
                lat.append(t.info.execution_time_ms / 1000)
            uso = t.info.token_usage or {}
            if uso:
                tok_in.append(uso.get("input_tokens", 0))
                tok_out.append(uso.get("output_tokens", 0))
            root = t.data.spans[0]
            q = root.inputs.get("question", "") if isinstance(root.inputs, dict) else ""
            corr = next((a for a in t.info.assessments if a.name == "correctness"), None)
            if corr is not None and corr.feedback is not None:
                por_tipo[tipos.get(q, "?")].append(_si(corr.feedback.value))

        nombre = r["tags.mlflow.runName"]
        print(f"\n{nombre}  (chunk={r.get('params.chunk_size')}, run_id={r['run_id'][:8]})")
        if lat:
            print(f"  latencia por pregunta: media {statistics.mean(lat):.1f} s, "
                  f"mediana {statistics.median(lat):.1f} s")
        if tok_in:
            print(f"  tokens por pregunta: entrada {statistics.mean(tok_in):.0f}, "
                  f"salida {statistics.mean(tok_out):.0f}")
        for tipo, vals in sorted(por_tipo.items()):
            print(f"  correctness {tipo:16s} {sum(vals)}/{len(vals)}")


if __name__ == "__main__":
    main()
