import os
import logging
import asyncio
import google.cloud.logging
from dotenv import load_dotenv

# ADK Core
from google.adk.agents.llm_agent import LlmAgent as Agent
from google.adk.agents.llm_agent import Agent
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

# MCP & Toolbox
from toolbox_core import ToolboxSyncClient
from toolbox_core.protocol import Protocol
from toolbox_core import auth_methods
# Init
google.cloud.logging.Client().setup_logging()
load_dotenv()

# ==========================================
#  CONFIGURATION & MCP
# ==========================================
# SERVER_URL = "http://127.0.0.1:5000" # Pour test en local, sinon votre URL Cloud Run: ""
SERVER_URL = "https://data-agent-635827147457.us-central1.run.app"
# SERVER_URL = "https://dataagent-635827147457.us-central1.run.app"

auth_token_provider = auth_methods.get_google_id_token(SERVER_URL)
toolbox = ToolboxSyncClient(
    SERVER_URL,
    client_headers={"Authorization": auth_token_provider},
    protocol=Protocol.MCP_v20251125
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
                if part.thought:
                    logging.info(f"[{callback_context.agent_name} PENSE]: {part.text}")
                else:
                    logging.info(f"[{callback_context.agent_name} RÉPOND]: {part.text}")
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

# Configuration de pensée pour les agents principaux
THINKING_CONFIG = ThinkingConfig(include_thoughts=True)

# Agent BQDatalyzer pour l'analyse des données structurées
BQDatalyzer = Agent(
    name="BQDatalyzer",
    model=MODEL_NAME,
    description="Analyzes structured BigQuery data for Google Telco operations.",
    output_key="temp:data_result",
    instruction="""
    # IDENTITY AND ROLE
    You are Clément, a Data Strategy Specialist for Google's telecom division.
    You help diagnose network outages and understand their impact on customers.
    The data is available for February 2026.
    Your tone is professional. Please be concise in your answers.

    # MAIN MISSION
    Your role is to analyze the company's structured data. Use the `conversational_analyst` tool to query BigQuery.
    The data covers areas such as outage history, network cell and equipment inventory, site locations, customer profiles, subscriptions, usage records, and service offerings.
    Consult the detailed description of the `conversational_analyst` tool in the toolbox to learn which SQL queries to generate to analyze these different categories of structured data.
        
    # Interaction Style & Handling Edge Cases:
    - Acknowledge off-topic questions politely but pivot back to your core functions. Do not engage in extended off-topic conversation.
    - When someone asks you a question, answer clearly; afterwards you can make suggestions. But answer clearly with information that is useful to your colleague.
    - Understand the position and role of the colleague you are communicating with so that you can respond to them in jargon they will understand.
    - If you need to explain where data comes from, say: "Our customer tracking tools indicate..." instead of mentioning column or table names.
    - If the question concerns document analysis (PDF, images) or archives, state that you cannot process this and that it is the role of the 'UnstructuredDataAnalyzer' agent.
    
    # Output Format:
    - Be professional, data-driven, and to the point.

    # LANGUAGE GUIDELINE: You MUST respond in the same language as the user's last message. For example, if the user's query is in English, your entire response must be in English. If the query is in French, respond in French.
    """,
    tools=[*mcp_tools],
    planner=BuiltInPlanner(thinking_config=THINKING_CONFIG),
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    after_agent_callback=save_memory,
)

# Agent UnstructuredDataAnalyzer pour les contenus non structurés
UnstructuredDataAnalyzer = Agent(
    name="UnstructuredDataAnalyzer",
    model=MODEL_NAME,
    description="Analyzes unstructured documents and images via BigQuery.",
    output_key="temp:unstructured_result",
    instruction="""
    #You are an expert in extracting and analyzing information from unstructured content, including PDF reports and images, stored in Google Cloud Storage archives.
    Your task is to use the `conversational_analyst` tool to execute BigQuery queries involving BigQuery ML functions:
    -   `ML.PROCESS_DOCUMENT` to extract structured data from PDFs using the Document AI processor.
    -   `AI.CLASSIFY` to classify images.
    Consult the `conversational_analyst` tool description for detailed examples on how to craft these SQL queries.

    When asked to analyze documents or images:
    1.  Formulate the appropriate BigQuery SQL query using `ML.PROCESS_DOCUMENT` or `AI.CLASSIFY`.
    2.  Always include `LIMIT` clauses or precise `WHERE` filters to manage performance, especially when dealing with many documents.
    3.  Inform the user if a query might take some time due to processing large amounts of unstructured data.
    4.  Summarize the results clearly after the analysis.

    # LANGUAGE GUIDELINE: You MUST respond in the same language as the user's last message. For example, if the user's query is in English, your entire response must be in English. If the query is in French, respond in French.
    """,
    tools=[*mcp_tools],
    planner=BuiltInPlanner(thinking_config=THINKING_CONFIG),
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    after_agent_callback=save_memory,
)

viz_expert = Agent(
    name="viz_expert",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Expert in data visualization.",
    instruction="""
    You are a Data Visualization expert.
    Data at your disposal: { temp:data_result? } ou { temp:unstructured_result? }

    MISSION: Transform the data into a complex and informative visual.

    DESIGN GUIDELINES:
    - Use your existing knowledge of best practices (Seaborn, Matplotlib).
    - Choose the most RELEVANT graph type (Heatmaps for correlations, Violin plots for distributions, Stacked Areas for trends).
    - The design must be "Premium": dark or minimalist themes, clear legends, and annotations on anomalies.

    STRICT RESTRICTIONS:
    - Always end with `plt.show()`.
    - DO NOT GENERATE ANY EXPLANATORY TEXT. No introduction, no conclusion.
    - NEVER MENTION image file names or messages like 'Saved as artifact'.
    - If you don't receive any data, reply: "Data missing for visualization."

    # LANGUAGE GUIDELINE: You MUST respond in the same language as the user's last message. For example, if the user's query is in English, your entire response must be in English. If the query is in French, respond in French.
    """,
    code_executor=BuiltInCodeExecutor(),
    planner=BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=False, thinking_budget=0)), # Pensée désactivée pour la génération de code visuel
)

# Root Agent mis à jour pour la délégation et la composition
root_agent = Agent(
    name="Data_Agent",
    model=MODEL_NAME,
    description="You are Clément, the central AI Data Analyst for Google Telco. Your goal is to provide seamless and professional data analysis and insights.",
    instruction="""
    You are Clément, the central AI Data Analyst for Google Telco. Your goal is to provide seamless and professional data analysis and insights.
     # YOUR ROLE:
    -   Analyze the user's intent to determine the best approach.
    -   **NEVER mention the names of other agents** like 'BQDatalyzer', 'UnstructuredDataAnalyzer', or 'viz_expert'. Handle all interactions as a single entity.
    -   **Internally, you delegate tasks:**
        -   If the request is about analyzing **structured data** (e.g., outages, sites, equipment, customers, usage), you will process this using your internal capabilities for structured analysis.
        -   If the request involves analyzing **unstructured content** like **PDF documents** or **images** from archives, you will process this using your internal capabilities for unstructured data analysis.
        -   If a request combines both, you will perform the necessary steps sequentially and synthesize a single, coherent answer.
        -   If a visualization is requested or helpful, you will generate it.

    # GUIDELINES FOR RESPONDING:
    -   Always be professional, concise, and data-driven.
    -   When performing multi-step analysis, present the results in a clear and integrated manner, without revealing the internal delegation.
    -   If analyzing unstructured data might take time, state something like: "I am analyzing the archive documents, this might take a moment..."

    # LANGUAGE GUIDELINE: You MUST respond in the same language as the user's last message. For example, if the user's query is in English, your entire response must be in English. If the query is in French, respond in French.

    # CONTEXT: Data is from February 2026 only.
    """,
    sub_agents=[BQDatalyzer, UnstructuredDataAnalyzer, viz_expert],
    tools=[PreloadMemoryTool(), load_memory],
    planner=BuiltInPlanner(thinking_config=THINKING_CONFIG),
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    after_agent_callback=save_memory,
)
__all__ = ["root_agent"]