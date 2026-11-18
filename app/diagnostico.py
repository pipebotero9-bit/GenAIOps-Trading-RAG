# app/diagnostico.py
"""
Diagnóstico de fallos (Parte 3.2): lista las preguntas de un run de evaluación
con su nota de correctness, las fuentes que trajo el retriever y el
razonamiento de los jueces clave. Sirve para clasificar cada fallo como de
recuperación, generación, dataset o juez.

Uso:
    python app/diagnostico.py              # último run del experimento rag_evaluation
    python app/diagnostico.py <run_id>
    python app/diagnostico.py --solo-fallos
"""

import os
import sys

import mlflow
from mlflow.entities import SpanType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.tracking import EVAL_EXPERIMENT

JUECES = ("correctness", "retrieval_relevance", "retrieval_groundedness")


def _valor(a):
    fb = getattr(a, "feedback", None)
    return getattr(fb, "value", None) if fb is not None else getattr(a, "value", None)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    solo_fallos = "--solo-fallos" in sys.argv
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    if args:
        run_id = args[0]
    else:
        runs = mlflow.search_runs(experiment_names=[EVAL_EXPERIMENT],
                                  order_by=["start_time DESC"], max_results=1)
        run_id = runs.iloc[0]["run_id"]
    run = mlflow.get_run(run_id)
    print(f"Run: {run.info.run_name} ({run_id})")
    print(f"Parámetros: prompt={run.data.params.get('prompt')} k={run.data.params.get('k')} "
          f"chunk={run.data.params.get('chunk_size')}\n")

    traces = mlflow.search_traces(locations=[run.info.experiment_id], run_id=run_id,
                                return_type="list")
    for i, t in enumerate(traces, 1):
        notas = {}
        razones = {}
        for a in t.info.assessments or []:
            if a.name in JUECES:
                v = _valor(a)
                notas.setdefault(a.name, []).append(v)
                razones.setdefault(a.name, a.rationale or "")
        corr = (notas.get("correctness") or [None])[0]
        fallo = str(corr).lower() not in ("yes", "true", "1")
        if solo_fallos and not fallo:
            continue

        pregunta = ""
        root = t.data.spans[0] if t.data.spans else None
        if root is not None and isinstance(root.inputs, dict):
            pregunta = root.inputs.get("question", "")

        fuentes = []
        for span in t.search_spans(span_type=SpanType.RETRIEVER):
            for doc in span.outputs or []:
                md = doc.get("metadata") or {}
                nombre = os.path.basename(md.get("source", "?")).split("_")[0]
                fuentes.append(f"{nombre} p.{(md.get('page') or 0) + 1}")

        rr = notas.get("retrieval_relevance") or []
        rr_si = sum(str(v).lower() in ("yes", "true", "1") for v in rr)
        print(f"[{i:02d}] {'FALLA' if fallo else 'ok   '} | {pregunta[:90]}")
        print(f"     trace_id: {t.info.trace_id}")
        print(f"     fuentes: {', '.join(fuentes) or '—'}  (relevantes: {rr_si}/{len(rr)})")
        print(f"     groundedness: {(notas.get('retrieval_groundedness') or ['—'])[0]}")
        if fallo:
            print(f"     correctness: {razones.get('correctness', '')[:300]}")
        print()


if __name__ == "__main__":
    main()
