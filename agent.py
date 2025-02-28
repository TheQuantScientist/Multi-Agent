import os
from typing import Literal
from langchain_groq import ChatGroq
from fastapi import FastAPI, Form
from fastapi.responses import ORJSONResponse
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Quoc Anh ver2 API",
    description="A simplified LangGraph demo",
    version="1.0",
    default_response_class=ORJSONResponse
)

# Get API keys and initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)
tools = [TavilySearchResults(max_results=2)]

# Define Pydantic models for structured output
class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]
    reason: str

class StepResponse(BaseModel):
    step: str
    role: str
    content: str
    timestamp: str

class ChatResponse(BaseModel):
    status: str
    original_query: str
    workflow_steps: list[StepResponse]
    final_answer: str
    completed_at: str

# Define workflow nodes
def supervisor_node(state: MessagesState):
    messages = [
        {"role": "system", "content": """
        You are a supervisor routing tasks to either:
        - Researcher: For gathering info or web searches
        - Coder: For coding or calculations
        - FINISH: If the query is already answerable
        Analyze the latest message and decide.
        """}
    ] + state["messages"]
    
    response = llm.with_structured_output(SupervisorDecision).invoke(messages)
    return {"messages": state["messages"] + [HumanMessage(content=response.reason, name="supervisor")], "next": response.next}

def researcher_node(state: MessagesState):
    messages = [
        {"role": "system", "content": "You are a researcher. Use search tools or knowledge to answer."}
    ] + state["messages"]
    result = llm.invoke(messages)  # Simplified, no tool binding for now
    return {"messages": state["messages"] + [HumanMessage(content=result.content, name="researcher")], "next": "FINISH"}

def coder_node(state: MessagesState):
    messages = [
        {"role": "system", "content": "You are a coder. Solve technical or calculation-based queries."}
    ] + state["messages"]
    result = llm.invoke(messages)
    return {"messages": state["messages"] + [HumanMessage(content=result.content, name="coder")], "next": "FINISH"}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {"researcher": "researcher", "coder": "coder", "FINISH": END}
)
builder.add_edge("researcher", END)
builder.add_edge("coder", END)
graph = builder.compile()

# Define API endpoint
@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(message: str = Form(...)):
    inputs = {"messages": [HumanMessage(content=message, name="user")], "next": "supervisor"}
    workflow_steps = []
    final_answer = ""

    # Run the workflow
    for output in graph.stream(inputs):
        for key, value in output.items():
            if "messages" in value:
                latest_message = value["messages"][-1]
                step = StepResponse(
                    step=key,
                    role=latest_message.name,
                    content=latest_message.content,
                    timestamp=datetime.now().isoformat()
                )
                workflow_steps.append(step)
    
    final_answer = next(
        (step.content for step in reversed(workflow_steps) if step.role in ["researcher", "coder"]),
        "No answer generated"
    )

    return ChatResponse(
        status="completed",
        original_query=message,
        workflow_steps=workflow_steps,
        final_answer=final_answer,
        completed_at=datetime.now().isoformat()
    )

@app.get("/")
async def home():
    return {"message": "Welcome to the Simplified Grok API! Use /chat/ endpoint to interact."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
