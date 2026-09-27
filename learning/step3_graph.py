import operator
from typing import Annotated, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

REGION = "us-east-1"
pro_llm = ChatBedrockConverse(model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                              region_name=REGION, max_tokens=250)
con_llm = ChatBedrockConverse(model="global.amazon.nova-2-lite-v1:0",
                              region_name=REGION, max_tokens=250)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the {side} debater on the motion: {topic}\n"
               "Write plain prose, no markdown, at most 60 words. "
               "If your opponent has spoken, rebut their latest point first. "
               "Only state facts you are confident are accurate."),
    ("human", "Debate so far:\n{transcript}\n\nYour turn."),
])

class DebateState(TypedDict):
    topic: str
    round: int
    max_rounds: int
    transcript: Annotated[list[str], operator.add]

def speak(side, llm, state):
    chain = prompt | llm | StrOutputParser()
    text = chain.invoke({
        "side": side,
        "topic": state["topic"],
        "transcript": "\n".join(state["transcript"]) or "(no arguments yet)",
    })
    return f"{side} (round {state['round']}): {text.strip()}"

def pro_node(state):
    return {"transcript": [speak("FOR", pro_llm, state)]}

def con_node(state):
    return {"transcript": [speak("AGAINST", con_llm, state)],
            "round": state["round"] + 1}

def next_step(state):
    return "pro" if state["round"] <= state["max_rounds"] else END

graph = StateGraph(DebateState)
graph.add_node("pro", pro_node)
graph.add_node("con", con_node)
graph.add_edge(START, "pro")
graph.add_edge("pro", "con")
graph.add_conditional_edges("con", next_step, ["pro", END])
app = graph.compile()

inputs = {"topic": "Airbus fly-by-wire design is safer than Boeing's.",
          "round": 1, "max_rounds": 3, "transcript": []}

for update in app.stream(inputs, stream_mode="updates"):
    for node, data in update.items():
        print(f"[{node}] {data['transcript'][-1]}\n")
