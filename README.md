# Chatbot RAG de investigación en trading intradía

Opero futuros MNQ y estoy construyendo TyrVwap, una estrategia intradía para la sesión de Nueva York: entro cuando el precio cruza el VWAP y dos medias móviles confirman la dirección. Cada decisión de diseño (dónde poner el stop, a qué hora operar, si agregar un filtro más) me llevaba a releer los mismos papers buscando qué se había probado y con qué evidencia.

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
| 2. Dataset de evaluación | 19 preguntas (9 directas, 4 multi-documento, 3 sin respuesta, 3 adversarias); primera evaluación en curso |
| 3. Experimentos | Pendiente |
| 4. Evaluadores propios | Pendiente |
| 5. Confianza en el juez | Pendiente |
| 6. Producción y CI | Pendiente |
| 7. Informe | Pendiente |

Los resultados y las decisiones van en `INFORME.md`.

## Limitaciones que ya identifiqué

- Las tablas de Lo, Mamaysky y Wang salen ilegibles porque el PDF es un escaneo, y las de Kempen y Maier-Paape son imágenes sin texto.
- El modelo recibe el texto de cada fragmento pero no el nombre del paper, así que al comparar autores puede atribuir mal.
- Ningún paper usa MNQ ni barras de ticks: el bot me dice qué se ha probado, pero la validación para mi estrategia la tengo que hacer con mis propios datos.

Juan Felipe Botero · Finanzas, EAFIT
