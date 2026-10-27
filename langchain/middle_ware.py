import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import SystemMessage,HumanMessage


groq_model = init_chat_model(
    model = "openai/gpt-oss-20b",
    model_provider="groq"
)

groq_model.invoke("hi")


agent = create_agent(
    model = groq_model,
    checkpointer= InMemorySaver(),
    middleware= [
        SummarizationMiddleware(
            model = groq_model,
            trigger= ("messages",10),
            keep = ("messages",4)
        )
    ]
)


config = {
    "configurable":{
        "thread_id" :"test-1"
    }
}

# Alternative test data
questions = [
    "What is 2+2?",
    "What is 10*5?",
    "What is 100/4?",
    "What is 15-7?",
    "What is 3*3?",
    "What is 4*4?",
]

for q in questions:
    response = agent.invoke({"messages":[HumanMessage(content=q)]},config=config)
    print(f"messages: {response}")
    print(f"lenthn of messages is : {len(response["messages"])}")

    