from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

load_dotenv()

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

memory = MemorySaver()

tools = [add, multiply, divide]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return { "messages" : [llm_with_tools.invoke([sys_msg] + state["messages"])] }

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)

builder.add_edge("tools", "assistant")

react_graph = builder.compile()

display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))


config = {"configurable": {"thread_id": "1"}}

messages1 = [HumanMessage(content="Add 1 and 2 and give us the final result")]
response = react_graph.invoke({"messages": messages1})

for m in response['messages']:
    m.pretty_print()


messages2 = [HumanMessage(content="Multiply the result by 2.")]
newResponse = react_graph.invoke({"messages": messages2})

for m in newResponse['messages']:
    m.pretty_print()    
