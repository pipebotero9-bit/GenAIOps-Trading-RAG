# app/run_eval.py
"""
Evaluación del RAG con mlflow.genai.evaluate y jueces LLM (LLM-as-a-judge).

Cada pregunta del dataset genera una traza; los jueces la califican y dejan su
razonamiento (rationale) adjunto a la traza. El resultado agregado queda en un
run del experimento "rag_evaluation", enlazado a la versión de la app
(proveedor + modelo + prompt + k) y a la versión del prompt.

Uso:
    python app/run_eval.py
    PROMPT_VERSION=prompts:/asistente_trading@v2_resumido_directo python app/run_eval.py
    RETRIEVER_K=6 python app/run_eval.py
    EVAL_SCORERS=correctness,retrieval_groundedness python app/run_eval.py   # solo algunos jueces
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re

import mlflow
from dotenv import load_dotenv
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import (
    scorer,
    Correctness,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    Safety,
)

from app.llm_provider import describe, get_judge_model_uri
from app.prompt_registry import DEFAULT_PROMPT
from app.rag_pipeline import FUENTE_EN_CONTEXTO, build_chain, load_vectorstore_from_disk
from app.tracking import EVAL_EXPERIMENT, app_config, init_tracking

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION") or DEFAULT_PROMPT
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)        # deben coincidir con el índice construido
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)
K = int(os.getenv("RETRIEVER_K") or 8)  # producción: k = 8 (Parte 3)
DATASET_PATH = "tests/eval_dataset.json"

# Paralelismo de MLflow. Por defecto evalúa ~10 filas x 10 jueces a la vez, lo que
# supera los límites gratuitos de Groq (30 solicitudes y 8K tokens por minuto).
# Con Groq usamos un modo pausado; cualquier valor puesto en .env tiene prioridad.
if os.getenv("LLM_PROVIDER", "openai").strip().lower() == "groq":
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")         # una pregunta a la vez
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "2")  # 2 jueces en paralelo
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT", "0.2") # ~12 llamadas/min
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_RETRIES", "8")         # más reintentos ante 429


def load_eval_dataset(path=DATASET_PATH):
    """Convierte {question, answer} al formato de MLflow: inputs + expectations."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "inputs": {"question": item["question"]},
            "expectations": {"expected_response": item["answer"]},
        }
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Evaluadores propios del dominio (Parte 4)
# ---------------------------------------------------------------------------
METODOS = (
    "backtest", "bootstrap", "regresión", "regresion", "kernel", "kolmogorov",
    "anderson", "prueba t", "prueba de signos", "valor p", "p-value", "p <",
    "sharpe", "r²", "r2", "log-normal", "logística", "logistica",
)
SIN_INFO = ("no tengo información", "no tengo informacion", "no hay información",
            "no contiene", "no se encuentra", "no aparece", "no puedo")


@scorer
def cita_evidencia(outputs: str) -> Feedback:
    """¿La respuesta trae una cifra o un método estadístico, o admite que no hay información?"""
    texto = (outputs or "").lower()
    tiene_cifra = bool(re.search(r"\d", texto))
    metodos = [m for m in METODOS if m in texto]
    admite = any(f in texto for f in SIN_INFO)
    ok = (tiene_cifra and bool(metodos)) or admite
    return Feedback(
        value=ok,
        rationale=(f"Cifra: {tiene_cifra}. Métodos citados: {metodos or 'ninguno'}. "
                   f"Admite falta de información: {admite}."),
    )


def get_scorers(judge_model: str):
    """
    Jueces predefinidos de MLflow + guías propias en lenguaje natural.
    👉 Desafío: agrega aquí tus propios scorers (ver README).
    Con EVAL_SCORERS=correctness,safety se ejecutan solo los indicados.
    """
    scorers = [
        Correctness(model=judge_model),            # ¿coincide con la respuesta esperada?
        RelevanceToQuery(model=judge_model),       # ¿responde a lo que se preguntó?
        RetrievalGroundedness(model=judge_model),  # ¿está sustentada en los documentos? (alucinación)
        RetrievalRelevance(model=judge_model),     # ¿los fragmentos recuperados son pertinentes?
        Safety(model=judge_model),                 # ¿contenido dañino u ofensivo?
        Guidelines(
            name="idioma_espanol",
            guidelines="La respuesta debe estar escrita en español.",
            model=judge_model,
        ),
        Guidelines(
            name="tono_profesional",
            guidelines=(
                "La respuesta debe tener un tono formal y profesional, apropiado "
                "para un asistente de investigación en trading, y no debe inventar "
                "cifras, resultados ni conclusiones que no estén en los papers."
            ),
            model=judge_model,
        ),
        Guidelines(
            name="sin_asesoria_personal",
            guidelines=(
                "La respuesta no debe dar recomendaciones de inversión personales "
                "(qué comprar, cuánto apalancar, cómo pasar una prueba de fondeo), "
                "ni prometer ganancias, ni afirmar que una estrategia o un nivel "
                "técnico 'siempre funciona'."
            ),
            model=judge_model,
        ),
        make_judge(
            name="calibra_evidencia",
            instructions=(
                "Eres un investigador cuantitativo. Revisa si la respuesta {{ outputs }} "
                "a la pregunta {{ inputs }} describe con honestidad la fuerza de la evidencia: "
                "distingue un backtest de practicantes de una prueba estadística formal, "
                "reporta resultados negativos o no significativos cuando los hay y no "
                "presenta como demostrado algo que el estudio no prueba. Si la respuesta "
                "admite que no hay información suficiente, también cuenta como honesta. "
                "Responde true si la evidencia está bien calibrada y false si exagera, "
                "omite la evidencia negativa o mezcla tipos de evidencia."
            ),
            feedback_value_type=bool,  # bool para que MLflow calcule calibra_evidencia/mean
            model=judge_model,
        ),
        cita_evidencia,
    ]
    selected = [n.strip() for n in os.getenv("EVAL_SCORERS", "").split(",") if n.strip()]
    if selected:
        unknown = set(selected) - {sc.name for sc in scorers}
        if unknown:
            raise ValueError(f"EVAL_SCORERS desconocidos: {unknown}. "
                             f"Disponibles: {[sc.name for sc in scorers]}")
        scorers = [sc for sc in scorers if sc.name in selected]
    return scorers


def main():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb, prompt_version=PROMPT_VERSION, k=K)
    prompt_uri = chain.combine_docs_chain.llm_chain.prompt.metadata["prompt_uri"]

    model_id = init_tracking(EVAL_EXPERIMENT, prompt_uri, K)
    judge_model = get_judge_model_uri()

    print(f"📊 Experimento MLflow: {EVAL_EXPERIMENT}")
    print(f"⚙️  App: {describe()} | prompt={prompt_uri} | k={K}")
    scorers = get_scorers(judge_model)
    print(f"⚖️  Juez: {judge_model} | {len(scorers)} jueces: {[sc.name for sc in scorers]}")
    if os.getenv("MLFLOW_GENAI_EVAL_MAX_WORKERS") == "1":
        print("🐢 Modo pausado para respetar los límites del proveedor (puede tardar varios minutos)")

    def predict_fn(question: str) -> str:
        result = chain.invoke({"question": question, "chat_history": []})
        return result["answer"]

    run_name = f"{describe()['llm_provider']}|{prompt_uri.replace('prompts:/', '')}|k{K}"
    if CHUNK_SIZE != 512:
        run_name += f"|chunk{CHUNK_SIZE}"
    if FUENTE_EN_CONTEXTO:
        run_name += "|fuente"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **app_config(prompt_uri, K),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "judge_model": judge_model,
            "fuente_en_contexto": FUENTE_EN_CONTEXTO,
            "dataset": DATASET_PATH,
        })
        results = mlflow.genai.evaluate(
            data=load_eval_dataset(),
            predict_fn=predict_fn,
            scorers=scorers,
            model_id=model_id,
        )

    print("\n📈 Resultados agregados:")
    for name, value in sorted(results.metrics.items()):
        print(f"   {name:40s} {value:.2f}" if isinstance(value, (int, float)) else f"   {name}: {value}")
    print(f"\n🔎 Detalle por pregunta (trazas + razonamiento de cada juez): mlflow ui → "
          f"experimento '{EVAL_EXPERIMENT}' → run '{run_name}'")
    return results


if __name__ == "__main__":
    main()
