# Informe final: chatbot RAG de investigación en trading intradía

Juan Felipe Botero · Finanzas, EAFIT · GenAIOps

Todas las cifras salen del experimento `rag_evaluation` en `mlflow.db`: 19 preguntas (9 directas, 4 multi-documento, 3 sin respuesta y 3 adversarias), bot `openai/gpt-oss-120b` en Groq y juez `openai:/gpt-4o-mini`. Las cifras por tipo, latencia y tokens las calculé con `app/resumen_informe.py` y el diagnóstico con `app/diagnostico.py`.

## 1. Configuración en producción

| Configuración | correctness | relevance | groundedness | retrieval relevance | tono | idioma |
|---|---|---|---|---|---|---|
| Base: prompt v1, k = 4, chunk 512 | 0,63 | 0,53 | 0,32 | 0,24 | 0,58 | 0,95 |
| Prompt v2 (corto) | 0,63 | 0,42 | 0,42 | 0,24 | 0,21 | 0,68 |
| k = 8 | **0,79** | 0,68 | 0,47 | 0,21 | 0,58 | 0,84 |
| Chunk 256 | 0,53 | 0,58 | 0,26 | 0,20 | 0,53 | 1,00 |
| k = 8 (repetición con 10 jueces) | 0,74 | 0,68 | 0,42 | 0,20 | 0,53 | 0,89 |
| **k = 8 + nombre del paper en el contexto** | **0,79** | **0,74** | 0,42 | 0,21 | **0,84** | **1,00** |

Safety fue 1,00 en todas. Llevé a producción **prompt v1, k = 8 y el nombre del paper antes de cada fragmento**. Frente a la base, correctness pasó de 0,63 a 0,79, relevance de 0,53 a 0,74, tono de 0,58 a 0,84 y mi juez `cita_evidencia` de 0,74 a 0,95.

Lo que explica cada cambio:

- **k = 8** fue la mejora principal. El problema era de búsqueda: con 4 fragmentos el correcto muchas veces no entraba. Con 8 entra más seguido, aunque cada fragmento extra sea menos relevante (por eso retrieval relevance baja un poco).
- **El prompt v2** no pide responder en español ni en tono profesional, y el modelo mezcló idiomas (0,68) y bajó el tono a 0,21.
- **Los fragmentos de 256** separan la cifra del método que la explica, y correctness cayó a 0,53.
- **El nombre del paper** nació del diagnóstico (sección 2): el modelo atribuía hallazgos de Howard a Zarattini y Aziz porque no sabía de qué documento venía cada fragmento.

Una advertencia: la misma configuración k = 8 dio 0,79 y 0,74 en dos corridas. Esa diferencia es una sola pregunta y viene del juez, así que no atribuyo a la fuente en contexto la mejora en correctness; sí las de tono (+0,31) y `cita_evidencia` (+0,21), que son grandes.

## 2. Qué preguntas cuestan más y por qué

| Tipo | Base | Producción |
|---|---|---|
| Directa | 6/9 | 8/9 |
| Multi-documento | **0/4** | **1/4** |
| Sin respuesta | 3/3 | 3/3 |
| Adversaria | 3/3 | 3/3 |

Las **multi-documento** son las más difíciles: la búsqueda trae fragmentos de un solo paper y el bot no puede comparar. Clasifiqué 5 fallos revisando sus trazas:

| Pregunta | Tipo de fallo | Evidencia en la traza |
|---|---|---|
| Reglas de entrada y stop del VWAP | Recuperación y generación | 6 de 8 fragmentos eran de Howard y ninguno de la página 8 del paper VWAP; el bot atribuyó a Zarattini el área de valor y las salidas por tiempo de Howard |
| Gao frente a Zarattini y Aziz | Recuperación | Ningún fragmento del paper VWAP; el bot lo admitió |
| ¿Coinciden Kempen y Tsinaslanidis? | Recuperación | 7 de 8 fragmentos de Tsinaslanidis; el de Kempen no traía la conclusión |
| Métodos de Lo y de Batchelor | Generación | Inventó "bootstrap y Monte Carlo implícitos" en Lo y omitió a Batchelor, aunque un fragmento suyo llegó |
| Operar la primera media hora (base) | Recuperación | Con k = 4 no llegó nada de Gao; con k = 8 se corrigió |

Falla más en **recuperación** (4 de 5). El juez de retrieval relevance se mantiene cerca de 0,2 porque califica cada fragmento contra toda la pregunta: en una pregunta de dos papers, un fragmento correcto de uno solo cuenta como irrelevante. Lo tomo como una limitación del juez para este tipo de pregunta.

## 3. Preguntas sin respuesta y adversarias

El bot **no inventó** en ninguna de las 3 preguntas sin respuesta (configuración óptima del VWAP en MNQ, ORB en Bitcoin, nivel del Nasdaq en 2027): en todas dijo que los papers no tienen esa información, en las seis configuraciones. También **resistió las 3 adversarias** (ignorar instrucciones, confirmar que el 61,8 % "siempre funciona" y una estrategia apalancada garantizada): safety 1,00 en todas las corridas, y en producción `sin_asesoria_personal` = 0,84 y `calibra_evidencia` aprobó 17 de 19 respuestas.

Mis tres evaluadores del dominio están en `get_scorers()` de `app/run_eval.py`:

| Evaluador | Tipo | Qué revisa |
|---|---|---|
| `cita_evidencia` | `@scorer` de código | Que la respuesta traiga una cifra y un método estadístico, o que admita que no hay información |
| `calibra_evidencia` | `make_judge` | Que distinga un backtest de una prueba estadística y no esconda la evidencia negativa |
| `sin_asesoria_personal` | `Guidelines` | Que no dé recomendaciones de inversión ni diga que algo "siempre funciona" |

## 4. Confianza en el juez

Califiqué a mano 10 respuestas de producción (4 directas, 3 multi-documento, 2 sin respuesta, 1 adversaria) con el criterio de correctness, sin ver al juez, y las registré en MLflow como `correctness_humano` con fuente HUMAN. Después las califiqué con un segundo juez, `gpt-4o`.

| Juez | Acuerdo conmigo | Desacuerdo |
|---|---|---|
| `gpt-4o-mini` | 9/10 (90 %) | Aprobó la pregunta de Lo et al. aunque el bot no dijo que la evidencia fue más fuerte en Nasdaq |
| `gpt-4o` | 9/10 (90 %) | Reprobó la de los stops de Howard porque no traía cifras, aunque la conclusión era correcta |

Empatan, pero se equivocan distinto: `gpt-4o-mini` es blando con respuestas incompletas y `gpt-4o` es estricto con las cifras. Mantengo `gpt-4o-mini` porque con 10 casos la diferencia no es concluyente, porque todos los experimentos se calificaron con él y porque `cita_evidencia` y `calibra_evidencia` cubren justamente las respuestas incompletas o sin evidencia. Confío en sus puntajes para comparar configuraciones, pero no en su valor absoluto: el 0,79 probablemente está un poco inflado en las preguntas de varias partes.

## 5. Costo de la configuración en producción

| Configuración | Tokens de entrada | Tokens de salida | Latencia media |
|---|---|---|---|
| Base (k = 4) | 787 | 397 | 1,3 s |
| Prompt v2 | 614 | 232 | 0,9 s |
| k = 8 | 1.220 | 514 | 1,6 s |
| **Producción (k = 8 + fuente)** | **1.423** | **593** | **1,7 s** |

Producción usa 1,8 veces los tokens de la base y tarda 0,4 s más por pregunta. Vale la pena: sube correctness de 0,63 a 0,79 y casi todas las respuestas pasan a citar evidencia, que es para lo que existe el bot. El costo que sí pesa es la cuota gratuita de Groq: cada evaluación con k = 8 gasta unos 40.000 tokens del bot, una quinta parte del límite diario.

## 6. Producción y quality gate

Promoví el prompt v1 a `@production` y dejé k = 8 y el nombre del paper como valores por defecto de la app y de la evaluación. En CI el prompt se toma por el alias del archivo, `prompts:/asistente_trading@v1_asistente_trading`. Los umbrales del gate en `tests/test_run_eval.py` son:

| Juez | Umbral | Justificación |
|---|---|---|
| correctness | 0,70 | Producción obtuvo 0,74-0,79 y el juez varía ±0,05; 0,70 tolera ese ruido y aun así bloquea la base (0,63) y el chunk 256 (0,53) |
| safety | 1,00 | Un asistente de trading no puede dar ninguna respuesta dañina |
| cita_evidencia | 0,85 | Producción obtuvo 0,95; sin el nombre del paper bajó a 0,74 |

Resultado del CI: _pendiente (se completa al correr `eval.yml` en `main` y en el PR con el prompt saboteado)._

## 7. Con una semana más

Atacaría primero la **recuperación en preguntas multi-documento**, que es donde más falla. Probaría generar una consulta por cada paper que mencione la pregunta (multi-query) y una búsqueda híbrida con BM25, porque las preguntas nombran autores que los embeddings no distinguen bien. Después cambiaría el modelo de embeddings, que admite pocos tokens por fragmento y puede ignorar el final de cada uno, y pasaría OCR a las tablas escaneadas de Lo et al. y a las imágenes de Kempen y Maier-Paape. Por último, ampliaría el dataset para que una pregunta no mueva el correctness en 0,05.
