import langchain
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain.messages import HumanMessage,SystemMessage,AIMessage
import os
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

@tool
def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"
@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_model = init_chat_model(model = "openai/gpt-oss-20b",model_provider="groq")
x = groq_model.invoke("hi")
x.content

agent = create_agent(
    model = groq_model,
    tools= [read_email_tool,send_email_tool],
    checkpointer= InMemorySaver(),
    middleware= [HumanInTheLoopMiddleware(
        interrupt_on = {
            "send_email_tool":{
            "allowed_decisions":["approve","edit","reject"]},
            "read_email_tool":False})])



config = {"configurable":{"thread_id":"test-3"}}

result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config
)

result["messages"][-1].content

"__interrupt__" in result

result["__interrupt__"]


if "__interrupt__" in result:
    print("pause aprroving")
    result = agent.invoke(
        Command(
            resume = {"decisions": [
                {
                    "type":"reject"
                }
            ]

            }
        ),
        config = config
    )

result["messages"][-1].content

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

agent1 = create_agent(
    model = groq_model,
    tools = [send_email_tool,read_email_tool],
    checkpointer= InMemorySaver(),
    middleware= [HumanInTheLoopMiddleware(
        interrupt_on= {
            "send_email_tool": {
                "allowed_decisions":["approve","edit","reject"]
            },
            "read_email_tool": False
        }
    )]
)

config = {"configurable":{"thread_id":"test-2"}}

result = agent1.invoke({"messages": [HumanMessage(content= "send message to snagu@gamil.com with tiltle hi andf body how are you")]},config = config)

result["messages"][-1].content

result["__interrupt__"]

if "__interrupt__" in result:
    print("wait approving mail")
    result = agent1.invoke(
        Command(
            resume= {
                "decisions": [{"type": "approve"}]}
        ),
        config=config
    )



result["messages"][-1].content





