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

###token size based middle ware
from langchain.tools import tool

@tool
def search_hotels(city: str) -> str:
    """Search hotels - returns long response to use more tokens."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""

agent = create_agent(
    model = groq_model,
    tools=[search_hotels],
    checkpointer= InMemorySaver(),
    middleware=[SummarizationMiddleware(
        model=groq_model,
        trigger= ("tokens",550),
        keep= ("tokens",250)
    )]
)

config = {"configurable":{
    "thread_id":"test-2"
}}

def token_counter(messges:str)->int:
    total_char = sum([len(str(m.content)) for m in messges])
    return total_char//4 #approximately 4 cchar is eqal to 1 token


# Run test
cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
        config=config
    )
    
    tokens = token_counter(response["messages"])
    print(f"{city}: ~{tokens} tokens, {len(response['messages'])} messages")
    print(f"{(response['messages'])}")

import langchain
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from langchain.messages import SystemMessage,AIMessage,HumanMessage
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


groq_model = init_chat_model(
    model = "openai/gpt-oss-20b",
    model_provider="groq"
)

groq_model.invoke("hi")



agent = create_agent(
    model=groq_model,
    middleware= [
        SummarizationMiddleware(
            model= groq_model,
            trigger= ("messages",10),
            keep= ("messages",5)
        )
    ],
    checkpointer= InMemorySaver()
)

config = {"configurable":{"thread_id":"test-1"}}

questions = [
    "what is attention"
]


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
