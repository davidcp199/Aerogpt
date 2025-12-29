import json
import logging
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_provider import llm_creative, paths_config
from utils.Predictor_RUL import predict_RUL

logger = logging.getLogger(__name__)

PROMPT_RUL_RESPONSE = ChatPromptTemplate.from_template(
    """
Eres un asistente experto en mantenimiento de motores aeronáuticos.
Se te entrega la siguiente información:
- Predicción RUL del motor: {predicted_RUL} ciclos
- Valores de sensores y configuraciones: {sensor_values}

Las columnas corresponden a:
- Configuraciones del motor:
  - `setting1`: Ajuste de flujo de aire del compresor
  - `setting2`: Ajuste de presión de combustible
  - `setting3`: Posición de la válvula de geometría variable
- Sensores:
  - `comp_pressure1` (s1): Presión en la etapa 1 del compresor
  - `comp_pressure2` (s2): Presión en la etapa 2 del compresor
  - `comp_temp1` (s3): Temperatura en la etapa 1 del compresor
  - `comp_temp2` (s4): Temperatura en la etapa 2 del compresor
  - `comp_vibration` (s5): Vibración del compresor
  - `hpc_pressure` (s6): Presión en la cámara de alta presión (HPC)
  - `hpc_temp` (s7): Temperatura en la cámara de alta presión
  - `fan_speed_core` (s8): Velocidad del fan del núcleo
  - `fan_speed_lpc` (s9): Velocidad del fan de baja presión (LPC)
  - `turbine_temp` (s10): Temperatura en la turbina de alta presión
  - `turbine_vibration` (s11): Vibración en la turbina
  - `fuel_flow` (s12): Flujo de combustible
  - `oil_pressure` (s13): Presión de aceite
  - `oil_temp` (s14): Temperatura de aceite
  - `exhaust_temp` (s15): Temperatura de gases de escape
  - `bleed_air` (s16): Flujo de bleed air
  - `core_speed` (s17): Velocidad del núcleo
  - `lpc_exit_temp` (s18): Temperatura a la salida del LPC
  - `hpc_exit_temp` (s19): Temperatura a la salida del HPC
  - `vibration_fan` (s20): Vibración del fan
  - `fuel_valve_pos` (s21): Posición de la válvula de combustible

Instrucciones:
1. Evalúa el nivel de desgaste del motor según RUL:
   - RUL > 80: "Desgaste bajo. Continuar operación normal."
   - RUL > 40: "Desgaste moderado. Programar inspección preventiva."
   - RUL > 20: "Desgaste significativo. Evaluar inspección avanzada."
   - RUL > 5 : "Riesgo elevado. Requiere monitorización constante."
   - RUL <=5 : "ALERTA CRÍTICA: Recomendada retirada inmediata del motor."

2. Detecta degradación asociada a sensores clave:
   - Temperatura / Compresor: `comp_temp1` (s3)
   - Presión HPC: `hpc_pressure` (s6)
   - Vibraciones: `comp_vibration` (s5)
   - Fan speed / núcleo: `fan_speed_core` (s8)
   - Fuel flow: `fuel_flow` (s12)

3. **Al redactar la respuesta**, no es necesario mencionar todos los sensores. Solo describe:
   - Los sensores que muestran degradación significativa
   - Los sensores más importantes para entender el estado del motor
   - Patrones relevantes para cada situación concreta

4. Traduce los patrones a modos de fallo probables usando lenguaje claro y conciso.

Formato de salida: texto profesional, sin JSON.
"""
)



def rul_action(state):
    """
    Acción del agente RUL:
    - Toma state.pre_rul_data (DataFrame)
    - Calcula RUL
    - Genera explicaciones
    """
    # print(">>>RUL")
    logger.info(">>> RUL AGENT")
    state.source = "RUL"
    state.emit("\n---> RUL AGENT", level="debug")

    try:
        # Comprobar que hay datos acumulados
        if state.pre_rul_data is None or len(state.pre_rul_data) == 0:
            state.messages.append(AIMessage(content="No hay datos para calcular el RUL. Añada datos de configuracion y sensores del motor primero."))
            state.emit("\nNo hay datos para calcular el RUL. Añada datos de configuracion y sensores del motor primero.", level="user")
            state.needs_followup = False
            state.next_agent = None
            return state

        df_user = state.pre_rul_data

        base_path = paths_config["paths"]["data_directory"]
        fd = state.modelo_seleccionado or "FD001"

        try:
            pred = predict_RUL(df_user, base_path, fd=fd)
        except Exception as e:
            logger.exception("Error en predict_RUL: %s", e)
            state.messages.append(AIMessage(content=f"Error al calcular la RUL: {e}"))
            state.emit(f"\nError al calcular la RUL: {e}", level="user")
            state.needs_followup = False
            state.next_agent = None
            return state

        predicted_RUL = pred.get("predicted_RUL", None)
        #print(f"----->Predicted RUL: {predicted_RUL}")
        state.emit(f"\nPredicted RUL: {predicted_RUL}", level="debug")
        if predicted_RUL is None:
            state.messages.append(AIMessage(content="El modelo no devolvió una predicción válida."))
            state.emit("\nEl modelo no devolvió una predicción válida.", level="user")
            state.needs_followup = False
            state.next_agent = None
            return state

        sensor_values = df_user.iloc[-1].to_dict()
        chain = PROMPT_RUL_RESPONSE | llm_creative
        text = chain.invoke({
            "predicted_RUL": predicted_RUL,
            "sensor_values": sensor_values
        }).content.strip()

        state.messages.append(AIMessage(content=text))
        state.update_memory("RUL", text)
        state.rul = {
            "predicted_RUL": predicted_RUL,
            "text": text
        }

        # Si el motor está crítico, ir a agente Criticidad
        if isinstance(predicted_RUL, (int, float)) and predicted_RUL < 20:
            state.needs_followup = True
            state.next_agent = "Criticidad"
        else:
            state.needs_followup = True
            state.next_agent = "Final"

        return state

    except Exception as e:
        logger.exception("Error en rul_action (RUL): %s", e)
        state.messages.append(AIMessage(content=f"Error interno del agente RUL: {e}"))
        state.emit(f"\nError interno del agente RUL: {e}", level="user")
        return state
