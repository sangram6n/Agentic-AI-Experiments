from google.adk.agents.llm_agent import Agent
from google.adk.models import LiteLlm
from dotenv import load_dotenv
import os
import datetime
from zoneinfo import ZoneInfo


load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

root_agent = Agent(
    model=LiteLlm(model="ollama/qwen3:8b"),
    name='new_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
