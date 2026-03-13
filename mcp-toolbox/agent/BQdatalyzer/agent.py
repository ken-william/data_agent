import os
import logging
import asyncio
import google.cloud.logging
from dotenv import load_dotenv

# ADK Core
from google.adk.agents.llm_agent import LlmAgent as Agent
from google.adk.agents import SequentialAgent, LoopAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.models import Gemini
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from google.genai.types import ThinkingConfig

# Mémoire
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, exit_loop
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

# MCP & Toolbox (L'asynchrone est ici)
from toolbox_core import ToolboxSyncClient
from toolbox_core.protocol import Protocol
from toolbox_core import auth_methods
# Init
google.cloud.logging.Client().setup_logging()
load_dotenv()

# ==========================================
#  CONFIGURATION & MCP
# ==========================================
SERVER_URL = "https://dataagent-635827147457.us-central1.run.app"

auth_token_provider = auth_methods.get_google_id_token(SERVER_URL)
toolbox = ToolboxSyncClient(
    SERVER_URL,
    client_headers={"Authorization": auth_token_provider},
    protocol=Protocol.TOOLBOX
)
mcp_tools = toolbox.load_toolset('my_bq_toolset')



# ==========================================
#  CALLBACKS (LOGS & MÉMOIRE)
# ==========================================
memory_service = InMemoryMemoryService()

def log_query_to_model(callback_context: CallbackContext, llm_request: LlmRequest):
    if llm_request.contents and llm_request.contents[-1].role == 'user':
        for part in llm_request.contents[-1].parts:
            if part.text:
                 logging.info(f"[REÇOIT]: {part.text}")

def log_model_response(callback_context: CallbackContext, llm_response: LlmResponse):
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                logging.info(f"[RÉPOND]: {part.text}")
            elif part.function_call:
                logging.info(f"[{callback_context.agent_name} APPELLE OUTIL]: {part.function_call.name}")

async def save_memory(callback_context: CallbackContext):
    if hasattr(callback_context, '_invocation_context'):
        await memory_service.add_session_to_memory(callback_context._invocation_context.session)

# ==========================================
#  LES AGENTS SPÉCIALISÉS
# ==========================================

MODEL_NAME = "gemini-3-flash-preview" 
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

BQDatalyzer = Agent(
    name="BQDatalyzer",
    model=MODEL_NAME,
    description="Analyste BigQuery spécialisé dans les données Telco Bouygues.",
    output_key="temp:data_result",
    instruction="""
    # IDENTITÉ ET RÔLE
    Tu es clément, spécialiste de la Stratégie Data pour l'entreprise de telecom Bouygues Telecom.
    Tu aides à diagnostiquer les pannes réseau et à comprendre l'impact sur les clients.
    Les données sont disponibles sur le mois de février 2026. 
    Ton ton est professionnel. sois concis dans tes réponses.

    # Interaction Style & Handling Edge Cases:
    - Acknowledge off-topic questions politely but pivot back to your core functions. Do not engage in extended off-topic conversation.
    - When someone asks you a question, answer clearly; afterwards you can make suggestions. But answer clearly with information that is useful to your colleague.
    - Understand the position and role of the colleague you are communicating with so that you can respond to them in jargon they will understand.
    - Si tu dois expliquer d'où vient une donnée, dis : "Nos outils de suivi clients indiquent..." au lieu de "La colonne churn_risk_probability contient... ou la table xxx contient".
    
    # Output Format:
    - Be professional, data-driven, and to the point.
    - 
    """,
    tools=[*mcp_tools],
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    after_agent_callback=save_memory,
)


viz_expert = Agent(
    name="viz_expert",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Expert visualisation de données.",
    instruction="""
    Tu es un expert en Data Visualization.
    Données à ta disposition : { temp:data_result? }

    MISSION : Transformer les données en un visuel complexe et informatif.
    
    CONSIGNES DE DESIGN :
    - Utilise ta connaissance native des meilleures pratiques (Seaborn, Matplotlib).
    - Choisis le type de graphique le plus PERTINENT (Heatmaps pour les corrélations, Violin plots pour les distributions, Stacked area pour l'évolution).
    - Le design doit être "Premium" : thèmes sombres ou épurés, légendes claires, annotations sur les anomalies.

    RESTRICTIONS STRICTES :
    - Termine impérativement par `plt.show()`.
    - NE GÉNÈRE AUCUN TEXTE d'explication. Pas d'introduction, pas de conclusion.
    - NE MENTIONNE JAMAIS le nom des fichiers images ou les messages de type 'Saved as artifact'.
    - Si tu ne reçois pas de données, réponds : "Données manquantes pour la visualisation."
    """,
    code_executor=BuiltInCodeExecutor(),
    planner=BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=False, thinking_budget=0)),
)



root_agent = Agent(
    name="Data_Agent",
    model=MODEL_NAME,
    description="Analyse et visualise les données réseau.",
    instruction="""
    Tu es Clément, le point d'entrée pour l'analyse Data de Bouygues Telecom.
    
    TON RÔLE :
    - Analyser l'intention de l'utilisateur.
    - utilise TOUJOURS 'conversational_analyst' par BQDatalyzer pour répondre aux questions des utilisateurs.
    - Si l'utilisateur veut voir un graphique, ou si après une analyse de chiffres un visuel semble utile : appelle 'viz_expert'.
    - Si l'utilisateur demande une analyse suivie d'un graphique, appelle les deux l'un après l'autre.
    - Ne répète pas les messages système du type 'Saved as artifact' ou 'transfer_to_agent' ou du code.
    - Sois toujours professionnel et concis.
    
    CONTEXTE : Données de février 2026 uniquement.
    """,
    sub_agents=[BQDatalyzer, viz_expert],
    tools=[PreloadMemoryTool(), load_memory],
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    after_agent_callback=save_memory,
)
__all__ = ["root_agent"]