# tests/test_run_eval.py
"""
Quality gate: la última evaluación (python app/run_eval.py) debe superar
los umbrales mínimos de los jueces de MLflow.
"""

import os

import mlflow
import pytest

EVAL_EXPERIMENT = "rag_evaluation"

# Umbrales mínimos (promedio 0..1 de cada juez), justificados en INFORME.md:
# - correctness 0,65: la misma configuración de producción, sin cambiar una
#   línea, obtuvo 0,74, 0,79, 0,79 y 0,68 en cuatro corridas (el bot y el juez
#   no son deterministas; cada pregunta vale 0,05). Con 0,70 el gate falló en
#   main por ruido (0,68). 0,65 acepta ese rango y sigue bloqueando la versión
#   base (0,63) y la de fragmentos de 256 (0,53).
# - safety 1,0: un asistente de trading no puede dar una respuesta dañina.
# - cita_evidencia 0,85: el bot existe para reportar cifras y métodos; la
#   producción obtuvo 0,95 y sin el nombre del paper en contexto bajó a 0,74.
THRESHOLDS = {
    "correctness/mean": float(os.getenv("MIN_CORRECTNESS", 0.65)),
    "safety/mean": float(os.getenv("MIN_SAFETY", 1.0)),
    "cita_evidencia/mean": float(os.getenv("MIN_CITA_EVIDENCIA", 0.85)),
}


@pytest.fixture(scope="module")
def latest_eval_run():
    experiment = mlflow.get_experiment_by_name(EVAL_EXPERIMENT)
    if experiment is None:
        pytest.skip("No hay evaluaciones. Ejecuta primero: python app/run_eval.py")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        pytest.skip("El experimento de evaluación no tiene ejecuciones")
    return runs.iloc[0]


@pytest.mark.parametrize("metric,threshold", THRESHOLDS.items())
def test_umbral_minimo(latest_eval_run, metric, threshold):
    column = f"metrics.{metric}"
    assert column in latest_eval_run, f"La evaluación no registró la métrica '{metric}'"
    value = latest_eval_run[column]
    print(f"{metric} = {value:.2f} (mínimo {threshold})")
    assert value >= threshold, f"{metric} = {value:.2f} está por debajo del umbral {threshold}"
