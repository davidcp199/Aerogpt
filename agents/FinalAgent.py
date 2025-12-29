import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from utils.llm_provider import llm_creative
from agents.State import AgentState

logger = logging.getLogger(__name__)

FINAL_AGENT_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente experto en aviación, certificación aeronáutica, normativa EASA/FAA
y mantenimiento aeronáutico.

Tu función es generar UNA RESPUESTA FINAL COMPLETA Y ADAPTADA A LA INTENCIÓN DEL USUARIO,
basándote exclusivamente en la información proporcionada por los agentes.

PRINCIPIO CLAVE
Antes de responder, debes razonar internamente:
- Qué tipo de pregunta hace el usuario: regulatoria, técnica, operativa o mixta.
- Qué información es relevante y cuál debe ignorarse por completo.
- Si hay datos disponibles de la fuente relevante, debes exponerlos casi completos para enriquecer la respuesta.
- Información secundaria (criticidad, RUL, reparaciones) se puede incluir solo si es coherente y aporta valor al contexto.

TIPOS DE PREGUNTA (clasificación implícita)
1. Regulación pura → definiciones, límites, certificación
2. Regulación aplicada → normativa con implicaciones operativas
3. Técnica descriptiva → funcionamiento de sistemas
4. Técnica con estado → análisis técnico con criticidad
5. Caso operacional completo → RUL + criticidad + acciones

JERARQUÍA DE FUENTES
- Regulación pura → Regulación (exclusiva)
- Regulación aplicada → Regulación (principal) + Criticidad/Acciones solo si aportan valor
- Técnica descriptiva → Técnico
- Técnica con estado → Técnico > Criticidad > Reparación
- Caso completo → RUL > Criticidad > Reparación > Regulación

REGLAS ESTRICTAS
1. NO inventes contexto operativo que el usuario no haya dado.
2. No incluyas información irrelevante o contradictoria que no aporte valor ni se corresponda con la intención de la pregunta.
3. Prioriza siempre la fuente más relevante según la jerarquía definida.
4. Si hay datos en la fuente prioritaria, preséntalos completos y estructurados para enriquecer la respuesta.
5. Mantén precisión normativa: CS-33/FAR33 para motores, CS-25/FAR25 para aeronaves, ORO/CAT/MEL/AFM según aplique.
6. La respuesta debe adaptarse a la pregunta: explicación, definición normativa, aclaración técnica o análisis operativo según corresponda.

FORMATO DE RESPUESTA
- Adáptalo a la pregunta.
- Incluye secciones solo si aportan valor.
- Prioriza la fuente más relevante y muestra su información casi completa.
- Información secundaria (criticidad, reparaciones, RUL) solo si es coherente con la intención de la pregunta.
- No propongas acciones si no han sido solicitadas explícitamente.

PREGUNTA DEL USUARIO
{user_question}

INFORMACIÓN DISPONIBLE DE AGENTES
Regulación: {regulation_info}
RUL: {rul_info}
Técnico: {technical_info}
Criticidad: {criticality_info}
Reparación: {repair_info}
""")




def final_action(state):
    """
    Acción del agente Final:
    - Integra información de otros agentes
    - Responde a la pregunta del usuario
    """
    # print(">>>FINAL Agent")
    state.emit("\n---> FINAL AGENT", level="debug")
    logger.info(">>> FINAL AGENT")
    state.source = "Final"

    try:
        user_question = state.messages[-1].content

        regulation_info = getattr(state, "regulation", None)
        criticality_info = getattr(state, "criticidad", None)
        repair_info = getattr(state, "reparacion", None)

        # RUL
        rul_state = getattr(state, "rul", None)
        if isinstance(rul_state, dict):
            rul_info = f"RUL estimado: {rul_state.get('predicted_RUL')} ciclos. {rul_state.get('text', '')}"
        else:
            rul_info = rul_state


        # TÉCNICO (CLAVE PARA EVITAR EL ERROR)
        technical_info = getattr(state, "tecnico", None)

        chain = FINAL_AGENT_PROMPT | llm_creative
        response = chain.invoke({
            "user_question": user_question,
            "regulation_info": regulation_info,
            "criticality_info": criticality_info,
            "repair_info": repair_info,
            "rul_info": rul_info,
            "technical_info": technical_info
        })

        print("==============================Entrando en FinalAgent, respuesta generada.=========================================")
        print(f"----> regulation_info: {regulation_info} \n")
        print(f"----> criticality_info: {criticality_info} \n")
        print(f"----> repair_info: {repair_info} \n")
        print(f"----> rul_info: {rul_info} \n")
        print(f"----> technical_info: {technical_info} \n")
        print("==============================SALIENDO en FinalAgent, respuesta generada.=========================================")

        state.messages.append(AIMessage(content=response.content.strip()))
        state.emit(response.content.strip(), level="user")
        state.needs_followup = False
        state.next_agent = None
        return state

    except Exception as e:
        logger.exception("Error interno en FinalAgent: %s", e)
        state.messages.append(
            AIMessage(content=f"Error generando respuesta final: {e}")
        )
        state.emit(f"\nError generando respuesta final: {e}", level="user")   
        state.needs_followup = False
        state.next_agent = None
        return state
