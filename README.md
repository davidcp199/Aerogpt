 AeroGPT
Asistente Inteligente para Análisis Aeronáutico Basado en Multi-Agentes

AeroGPT es un sistema avanzado de análisis aeronáutico que utiliza inteligencia artificial multi-agente para proporcionar respuestas especializadas sobre regulaciones, criticidad de componentes, reparaciones, análisis técnico y predicción de vida útil remanente (RUL) de motores aeronáuticos.

🌟 Características Principales
Sistema Multi-Agente Especializado: Arquitectura basada en LangGraph con 9 agentes especializados que trabajan de forma coordinada GraphBuilder.py:34-60

RAG (Retrieval Augmented Generation): Consulta de bases de datos vectoriales con documentación técnica y regulatoria aeronáutica paths.yaml:45-49

Predicción RUL: Análisis de vida útil remanente basado en datos CMAPSS con modelos de deep learning extract_cmapss.py:46-48

Interfaz Streamlit Intuitiva: UI moderna con historial de conversación y visualización de decisiones internas app.py:20-27

Instalación Automatizada: Sistema completamente productivizado con setup automático de dependencias run.py:86-89

🏗️ Arquitectura del Sistema
El sistema está construido sobre una arquitectura de grafos de estados con los siguientes agentes especializados:


Agentes Especializados
Agente	Función
Supervisor	Analiza la consulta y rutea al agente apropiado GraphBuilder.py:17-22
PreRUL	Prepara y valida datos para análisis RUL
RUL	Calcula predicciones de vida útil remanente
Criticidad	Evalúa la criticidad de componentes y MEL/CDL
Reparación	Proporciona procedimientos y mejores prácticas de reparación
Regulación	Consulta normativas FAA, EASA y documentación regulatoria
Técnico	Análisis técnico detallado de sistemas aeronáuticos
General	Responde consultas generales sobre aviación
Final	Consolida y presenta la respuesta final al usuario GraphBuilder.py:38-46
📋 Requisitos Previos
Python 3.8 o superior (recomendado 3.10.13) run.py:16-21
OpenAI API Key configurada
Sistema operativo Windows (el script usa rutas Windows para el entorno virtual)
🚀 Instalación y Configuración
Paso 1: Clonar el Repositorio
git clone https://github.com/davidcp199/Aerogpt.git  
cd Aerogpt
Paso 2: Configurar API Key
Crea un archivo .env en la raíz del proyecto con tu API key de OpenAI:

OPENAI_API_KEY=tu-api-key-aqui
run.py:71-76

Paso 3: Ejecutar el Sistema
python run.py
¡Eso es todo! El script run.py se encarga automáticamente de:

✅ Verificar la versión de Python
✅ Crear un entorno virtual llamado AEROGPT_ENV
✅ Instalar todas las dependencias desde config/requirements.yaml
✅ Verificar la existencia del archivo .env
✅ Lanzar la interfaz de Streamlit automáticamente run.py:23-31
La interfaz se abrirá automáticamente en tu navegador predeterminado y estará lista para usar.

💻 Uso
Interfaz Streamlit
Una vez iniciada la aplicación, podrás:

Realizar consultas en lenguaje natural sobre temas aeronáuticos
Ver decisiones internas del sistema multi-agente (activando la opción en el sidebar) app.py:62-65
Consultar historial por agente para análisis detallado app.py:67-70
Limpiar conversación para comenzar un nuevo caso app.py:72-85
Modo Consola (Opcional)
También puedes ejecutar el sistema en modo consola:

python main.py
Comandos disponibles:

stop - Salir del programa
show history - Ver historial por agente
show conversation - Ver resumen de conversación main.py:76-99
📁 Estructura del Proyecto
Aerogpt/  
├── agents/                    # Agentes especializados  
│   ├── SupervisorAgent.py    # Agente coordinador  
│   ├── RulAgent.py           # Predicción de vida útil  
│   ├── CriticidadAgent.py    # Análisis de criticidad  
│   ├── ReparacionAgent.py    # Procedimientos de reparación  
│   ├── RegulacionAgent.py    # Consultas regulatorias  
│   ├── TecnicoAgent.py       # Análisis técnico  
│   ├── GeneralAgent.py       # Consultas generales  
│   ├── PreRulAgent.py        # Preparación de datos RUL  
│   ├── FinalAgent.py         # Consolidación de respuestas  
│   ├── GraphBuilder.py       # Constructor del grafo  
│   └── State.py              # Definición del estado  
├── config/                    # Archivos de configuración  
│   ├── requirements.yaml     # Dependencias del sistema  
│   └── paths.yaml           # Rutas de datos y modelos  
├── data/                      # Datos del sistema  
│   ├── raw/                  # Datos crudos (CMAPSS, EASA, FAA, etc.)  
│   ├── processed/            # Datos procesados  
│   └── vectorstores/         # Bases de datos vectoriales RAG  
├── rag/                       # Sistema RAG  
│   └── ingest/               # Scripts de ingesta de datos  
├── tools/                     # Herramientas especializadas  
│   ├── extract_cmapss.py    # Extracción de datos CMAPSS  
│   └── tool_output_to_df.py # Conversión a DataFrame  
├── utils/                     # Utilidades  
├── app.py                     # Interfaz Streamlit  
├── main.py                    # Ejecución en consola  
└── run.py                     # Script de instalación y ejecución  
paths.yaml:1-49

🛠️ Tecnologías Utilizadas
Framework Principal
LangChain (1.1.3): Framework para aplicaciones con LLMs requirements.yaml:19-25
LangGraph: Construcción de grafos de estados para agentes
Streamlit (1.52.2): Interfaz de usuario web
Modelos y AI
OpenAI GPT (2.8.1): Modelo de lenguaje principal
FAISS (1.13.1): Búsqueda vectorial para RAG
PyTorch (2.5.1): Framework de deep learning
Transformers (4.57.3): Modelos de HuggingFace
Análisis de Datos
NumPy (2.1.2), Pandas (2.2.3): Manipulación de datos
Scikit-learn (1.6.1): Machine learning
Matplotlib (3.9.2), Seaborn (0.13.2): Visualización requirements.yaml:3-29
🔄 Sistema de Estado
El sistema mantiene un estado compartido entre todos los agentes que incluye:

Mensajes: Historial de la conversación
Decisiones: Ruteo entre agentes
Datos técnicos: Regulaciones, criticidad, reparaciones, datos RUL
Historial por agente: Trazabilidad completa de decisiones
Buffers de salida: Separación entre respuestas de usuario y debug State.py:7-47
📊 Funcionalidades Avanzadas
Detección de Nuevo Caso
El sistema detecta automáticamente cuando se inicia un nuevo caso y limpia el contexto técnico previo, manteniendo solo el historial conversacional relevante. app.py:99-103

Memoria Conversacional
Mantiene un resumen de conversación limitado a 3000 caracteres para contexto eficiente. State.py:55-70

Sistema de Emisión Dual
Output Buffer: Respuestas para el usuario
Debug Buffer: Decisiones internas del sistema State.py:73-81
📝 Notas Adicionales
Sistema Productivizado: Listo para uso en producción con manejo robusto de errores
Entorno Virtual Automático: No requiere configuración manual de dependencias
Logging Integrado: Sistema completo de logging para debugging y monitoreo main.py:28-36
Extensible: Arquitectura modular que permite agregar nuevos agentes fácilmente
RAG Personalizado: Bases de datos vectoriales con documentación aeronáutica específica
