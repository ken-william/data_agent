
import os
import logging
import subprocess
import google.cloud.logging
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token

from dotenv import load_dotenv

from google.adk.agents.llm_agent import LlmAgent as Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from toolbox_core import ToolboxSyncClient
from toolbox_core.protocol import Protocol
from toolbox_core import auth_methods

# Initialisation des logs Cloud
google.cloud.logging.Client().setup_logging()
load_dotenv()

# ==========================================
# 1. CONFIGURATION & AUTHENTIFICATION
# ==========================================
SERVER_URL = "https://data-agent-635827147457.us-central1.run.app"

auth_token_provider = auth_methods.get_google_id_token(SERVER_URL)
toolbox = ToolboxSyncClient(SERVER_URL,client_headers={"Authorization": auth_token_provider},protocol=Protocol.TOOLBOX)
mcp_tools = toolbox.load_toolset('my_bq_toolset')

# ==========================================
# CALLBACKS
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
# AGENT PRINCIPAL (Data Agent)
# ==========================================
root_agent = Agent(
  name="Data_Agent",
  model="gemini-3-flash-preview",
  description="Agent spécialisé dans la Stratégie Data chez Ecommarket",
  instruction="""
    # IDENTITÉ ET RÔLE
    Tu es clément, spécialiste de la Stratégie Data chez Ecommarket. Ecommarket est une entreprise de vente.
    Ton ton est professionnel. sois concis dans tes réponses.

    # Interaction Style & Handling Edge Cases:
    - Acknowledge off-topic questions politely but pivot back to your core functions. Do not engage in extended off-topic conversation.
    - When someone asks you a question, answer clearly; afterwards you can make suggestions. But answer clearly with information that is useful to your colleague.
    
    # Output Format:
    - Be professional, data-driven, and to the point.

    """,
  tools=[*mcp_tools,PreloadMemoryTool(), load_memory],
  before_model_callback=log_query_to_model,
  after_model_callback=log_model_response,
  after_agent_callback=save_memory,
)