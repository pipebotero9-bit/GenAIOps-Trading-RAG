# Chatbot RAG de investigación en trading intradía

Opero futuros y estoy construyendo una estrategia intradía para la sesión de Nueva York.  Cada decisión de diseño (dónde poner el stop, a qué hora operar, si agregar un filtro más) me llevaba a releer los mismos papers buscando qué se había probado y con qué evidencia.

Para el proyecto final de GenAIOps convertí esa búsqueda en un chatbot. Le pregunto, por ejemplo, si un stop por precio destruye valor o si los retrocesos de Fibonacci funcionan, y me responde con el método, la muestra y las cifras del paper, separando un backtest de practicantes de una prueba estadística.

Partí del repositorio de clase ([darkanita/GenAIOps_Pycon2025](https://github.com/darkanita/GenAIOps_Pycon2025)), que respondía preguntas de Recursos Humanos, y mantuve su arquitectura: LangChain, FAISS, MLflow, Streamlit y GitHub Actions. Cambié los documentos, los prompts, el dataset de evaluación y los jueces.

## Las preguntas que me interesan

| Pregunta de mi estrategia | Papers que la abordan |
|---|---|
| ¿Qué tan sólida es la evidencia detrás de operar con el VWAP? | Zarattini y Aziz (2023) |
| ¿Opero la primera media hora o la evito? | Gao, Han, Li y Zhou (2015); Howard (2026); Zarattini, Barbon y Aziz (2024) |
| ¿Mi stop por ATR ayuda o corta las ganancias? | Howard (2026) |
| ¿Vale la pena agregar retrocesos de Fibonacci? | Batchelor y Ramyar (2006); Tsinaslanidis, Guijarro y Voukelatos (2022); Kempen y Maier-Paape (2016) |
| ¿Cómo valido si un patrón o un filtro aporta información? | Lo, Mamaysky y Wang (2000) |

## Documentos

| Paper | Tipo de evidencia |
|---|---|
| Zarattini y Aziz (2023), *VWAP: The Holy Grail for Day Trading Systems* | Backtest de practicantes |
| Zarattini, Barbon y Aziz (2024), *A Profitable Day Trading Strategy for the U.S. Equity Market* | Backtest de practicantes |
| Gao, Han, Li y Zhou (2015), *Intraday Momentum: The First Half-Hour Return Predicts the Last Half-Hour Return* | Regresiones predictivas |
| Lo, Mamaysky y Wang (2000), *Foundations of Technical Analysis* | Regresión kernel, bondad de ajuste, Kolmogorov-Smirnov |
| Kempen y Maier-Paape (2016), *Survey on log-normally distributed market-technical trend data* | Ajuste log-normal, Anderson-Darling |
| Batchelor y Ramyar (2006), *Magic Numbers in the Dow* | Bootstrap estacionario por bloques |
| Tsinaslanidis, Guijarro y Voukelatos (2022), *Automatic identification and evaluation of Fibonacci retracements* | Regresión logística con bootstrap |
| Howard (2026), *Intraday Value Area Breakouts: Evidence from E-mini S&P 500 Futures* | Pruebas t con errores agrupados por día, prueba de signos |

Están en `data/pdfs/`: 273 páginas que se convierten en 1.234 fragmentos.

Incluí a propósito papers con evidencia negativa (los tres de Fibonacci concluyen que esos niveles no se distinguen del azar). Así puedo evaluar si el bot reporta lo que dice el documento o lo que yo quisiera oír.

## Estado

| Parte del desafío | Estado |
|---|---|
| 1. Cambio de dominio | Listo |
| 2. Dataset de evaluación | 19 preguntas (9 directas, 4 multi-documento, 3 sin respuesta, 3 adversarias). Primera evaluación: correctness 0,63, retrieval relevance 0,24, safety 1,00 |
| 3. Experimentos | 5 configuraciones y 5 fallos diagnosticados. La mejor: k = 8 con el nombre del paper en cada fragmento (correctness 0,79, tono 0,84) |
| 4. Evaluadores propios | `cita_evidencia` (código), `calibra_evidencia` (juez LLM) y `sin_asesoria_personal` (guía) |
| 5. Confianza en el juez | Califiqué 10 respuestas a mano: gpt-4o-mini y gpt-4o coinciden conmigo en 9 de 10 (90 %). Mantengo gpt-4o-mini |
| 6. Producción y CI | v1 en producción con k = 8 y nombre del paper. CI verde en `main` (correctness 0,74) y rojo en el PR con el prompt saboteado (0,47) |
| 7. Informe | Listo en `INFORME.md` |

Los resultados y las decisiones van en `INFORME.md`.

## Limitaciones que ya identifiqué

- Las tablas de Lo, Mamaysky y Wang salen ilegibles porque el PDF es un escaneo, y las de Kempen y Maier-Paape son imágenes sin texto.
- El buscador trae pocos fragmentos útiles (retrieval relevance 0,23 con k = 8): es la causa de la mayoría de los errores.
- Ningún paper usa MNQ ni barras de ticks: el bot me dice qué se ha probado, pero la validación para mi estrategia la tengo que hacer con mis propios datos.

Juan Felipe Botero · Finanzas, EAFIT
