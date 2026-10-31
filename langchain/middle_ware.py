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


import langchain
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage,HumanMessage,SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model



load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# groq_model = ChatGroq(
#     model = "openai/gpt-oss-20b"
# )

groq_model = init_chat_model(
    model = "openai/gpt-oss-20b",
    model_provider="groq"
)


groq_model.invoke("hi")


@tool
def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"
@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model=groq_model,
    tools = [send_email_tool,read_email_tool],
    middleware= [
        HumanInTheLoopMiddleware(
            interrupt_on= {
                "send_email_tool": {
                    "allowed_decisions" :["approve","edit","reject"]
                },
            "read_email_tool":False}
        )
    ],
    checkpointer= InMemorySaver(),
    system_prompt=(
        "You are a helpful assistant. Once a tool has been executed and returns "
        "a confirmation, the user's request is complete. Even if a human reviewer "
        "edited the tool arguments (such as changing the subject or body), do NOT "
        "call the tool again—simply confirm the action to the user.")

)
import uuid

# Generate a fresh thread_id so previous messages don't pollute the state
config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}




response = agent.invoke(
    {
        "messages":[HumanMessage(
            content=  "send emal to sangu5n@gmail.com with subject hi and content helo")]
    },
    config = config
)

if "__interrupt__" in response:
    response = agent.invoke(
        Command(
            resume= {
                "decisions" : [{
                    "type":"edit",
                    "edited_action":{
                        "name": "send_email_tool",
                        "args":{
                            "recipient":"nngu5n@gmail.com",
                            "subject":"hi sangu",
                            "body":"hello are you fine " }
                    }
                }
                ]
            }
        ),
        config=config
    )


response["messages"][-1].content
# Fix typo: 'responses' -> 'response'
print("Final AI Response:")
print(response["messages"][-1].content)

print("\n--- Full Message History Inspection ---")
for i, msg in enumerate(response["messages"]):
    print(f"[{i}] {msg.__class__.__name__}:")
    if getattr(msg, "tool_calls", None):
        print(f"    Tool Calls: {msg.tool_calls}")
    if msg.content:
        print(f"    Content: {msg.content}")