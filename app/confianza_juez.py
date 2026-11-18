# app/confianza_juez.py
"""
Parte 5: ¿cuánto coincide el juez de correctness con una persona?

Paso 1 — exportar 10 respuestas de la versión de producción para calificarlas a mano:
    python app/confianza_juez.py exportar
  Crea tests/calificacion_humana.md (para leer) y tests/calificacion_humana.csv
  (para llenar la columna correctness_humano con si/no). No muestra la nota del juez.

Paso 2 — registrar la calificación humana, calificar con un segundo juez y medir el acuerdo:
    python app/confianza_juez.py comparar openai:/gpt-4o
  Registra correctness_humano en cada traza (fuente HUMAN), califica las mismas 10
  respuestas con el segundo juez y guarda tests/confianza_juez.csv.
"""

import csv
import json
import os
import sys

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.genai.scorers import Correctness

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import app.llm_provider  # noqa: F401  (carga .env: claves de los jueces)
from app.tracking import EVAL_EXPERIMENT

DATASET = "tests/eval_dataset.json"
MD = "tests/calificacion_humana.md"
CSV = "tests/calificacion_humana.csv"
RESULTADOS = "tests/confianza_juez.csv"
CUOTA = {"directa": 4, "multi-documento": 3, "sin-respuesta": 2, "adversaria": 1}
JUEZ_1 = "correctness"


def _si(v):
    return str(v).strip().lower() in ("yes", "true", "si", "sí", "1")


def _run_produccion():
    if os.getenv("RUN_ID"):
        return mlflow.get_run(os.environ["RUN_ID"])
    d = mlflow.search_runs(experiment_names=[EVAL_EXPERIMENT],
                           filter_string="params.fuente_en_contexto = 'True' and params.k = '8'",
                           order_by=["start_time DESC"], max_results=1)
    return mlflow.get_run(d.iloc[0]["run_id"])


def _traces(run):
    traces = mlflow.search_traces(locations=[run.info.experiment_id], run_id=run.info.run_id,
                                  return_type="list")
    salida = {}
    for t in traces:
        root = t.data.spans[0]
        pregunta = (root.inputs or {}).get("question", "") if isinstance(root.inputs, dict) else ""
        resp = t.data.response
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except ValueError:
                pass
        if isinstance(resp, dict):
            resp = resp.get("answer", resp)
        juez = next((a for a in t.info.assessments if a.name == JUEZ_1), None)
        salida[pregunta] = {"trace_id": t.info.trace_id, "respuesta": str(resp), "juez": juez}
    return salida


def exportar():
    run = _run_produccion()
    dataset = json.load(open(DATASET, encoding="utf-8"))
    trazas = _traces(run)
    elegidas, cuenta = [], {k: 0 for k in CUOTA}
    for q in dataset:
        tipo = q.get("type", "directa")
        if cuenta.get(tipo, 99) < CUOTA.get(tipo, 0) and q["question"] in trazas:
            cuenta[tipo] += 1
            elegidas.append((q, trazas[q["question"]]))

    with open(MD, "w", encoding="utf-8") as f:
        f.write(f"# Calificación humana de correctness\n\nRun: `{run.info.run_name}`\n\n")
        f.write("Criterio: **sí** si la respuesta del bot contiene los hechos clave de la "
                "respuesta esperada y no la contradice. En las preguntas sin respuesta, **sí** si "
                "el bot admite que no tiene la información. En las adversarias, **sí** si se "
                "comporta como dice la respuesta esperada.\n\n")
        for n, (q, t) in enumerate(elegidas, 1):
            f.write(f"---\n\n## {n}. [{q['type']}] {q['question']}\n\n")
            f.write(f"**Respuesta esperada:** {q['answer']}\n\n")
            f.write(f"**Respuesta del bot:**\n\n{t['respuesta']}\n\n")
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "tipo", "trace_id", "pregunta", "correctness_humano"])
        for n, (q, t) in enumerate(elegidas, 1):
            w.writerow([n, q["type"], t["trace_id"], q["question"][:70], ""])
    print(f"Listo: {len(elegidas)} respuestas. Lee {MD} y llena la última columna de {CSV} con si/no.")


def comparar():
    juez_2 = sys.argv[2] if len(sys.argv) > 2 else "openai:/gpt-4o"
    autor = os.getenv("EVALUADOR_HUMANO", "juan_felipe_botero")
    dataset = {q["question"]: q for q in json.load(open(DATASET, encoding="utf-8"))}
    filas = list(csv.DictReader(open(CSV, encoding="utf-8")))
    faltan = [f["n"] for f in filas if f["correctness_humano"].strip() == ""]
    if faltan:
        sys.exit(f"Faltan calificaciones humanas en las filas: {', '.join(faltan)}")

    juez2 = Correctness(model=juez_2)
    resultados = []
    for f in filas:
        t = mlflow.get_trace(f["trace_id"])
        root = t.data.spans[0]
        pregunta = root.inputs.get("question", "")
        resp = t.data.response
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except ValueError:
                pass
        if isinstance(resp, dict):
            resp = resp.get("answer", resp)
        esperado = dataset[pregunta]["answer"]

        humano = _si(f["correctness_humano"])
        mlflow.log_feedback(
            trace_id=f["trace_id"], name="correctness_humano", value=humano,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=autor),
        )
        j1 = next((a for a in t.info.assessments if a.name == JUEZ_1), None)
        v1 = _si(j1.feedback.value) if j1 else None
        fb2 = juez2(inputs={"question": pregunta}, outputs=str(resp),
                    expectations={"expected_response": esperado})
        v2 = _si(fb2.value)
        mlflow.log_feedback(
            trace_id=f["trace_id"], name=f"correctness_{juez_2.split('/')[-1]}", value=v2,
            rationale=fb2.rationale,
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=juez_2),
        )
        resultados.append({
            "n": f["n"], "tipo": f["tipo"], "pregunta": f["pregunta"], "humano": humano,
            "juez_1": v1, "juez_2": v2,
            "razon_juez_1": (j1.rationale or "")[:400] if j1 else "",
            "razon_juez_2": (fb2.rationale or "")[:400],
        })

    with open(RESULTADOS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resultados[0]))
        w.writeheader()
        w.writerows(resultados)

    n = len(resultados)
    a1 = sum(r["humano"] == r["juez_1"] for r in resultados)
    a2 = sum(r["humano"] == r["juez_2"] for r in resultados)
    print(f"\nJuez 1 (el de la evaluación): acuerdo con humano {a1}/{n} = {a1 / n:.0%}")
    print(f"Juez 2 ({juez_2}): acuerdo con humano {a2}/{n} = {a2 / n:.0%}\n")
    print("Desacuerdos:")
    for r in resultados:
        if r["humano"] != r["juez_1"] or r["humano"] != r["juez_2"]:
            print(f"  [{r['n']}] {r['tipo']}: humano={r['humano']} juez1={r['juez_1']} "
                  f"juez2={r['juez_2']} | {r['pregunta']}")
    print(f"\nDetalle en {RESULTADOS}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    modo = sys.argv[1] if len(sys.argv) > 1 else "exportar"
    {"exportar": exportar, "comparar": comparar}[modo]()
