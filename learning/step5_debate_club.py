import os
os.environ["HF_HUB_OFFLINE"] = "1"

import operator
from typing import Annotated, Literal, TypedDict

import torch
from pydantic import BaseModel, Field
from transformers import pipeline
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

REGION = "us-east-1"
pro_llm = ChatBedrockConverse(model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                              region_name=REGION, max_tokens=250)
con_llm = ChatBedrockConverse(model="global.amazon.nova-2-lite-v1:0",
                              region_name=REGION, max_tokens=250)
judge_llm = ChatBedrockConverse(model="global.anthropic.claude-sonnet-4-6",
                                region_name=REGION, max_tokens=1500)

device = "mps" if torch.backends.mps.is_available() else "cpu"
tagger = pipeline("zero-shot-classification",
                  model="facebook/bart-large-mnli", device=device)
TAG_LABELS = {
    "emotional": "This text tries to make the reader feel fear or worry.",
    "rebuttal":  "This text directly responds to a claim made by an opponent.",
}

def tag(text):
    r = tagger(text, candidate_labels=list(TAG_LABELS.values()),
               multi_label=True, hypothesis_template="{}")
    scores = dict(zip(r["labels"], r["scores"]))
    return {k: round(scores[v], 2) for k, v in TAG_LABELS.items()}

def fmt(transcript, with_tags=False):
    if not transcript:
        return "(no arguments yet)"
    lines = []
    for t in transcript:
        line = f"{t['side']} (round {t['round']}): {t['text']}"
        if with_tags:
            line += f"  [style tags: {t['tags']}]"
        lines.append(line)
    return "\n".join(lines)

debate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the {side} debater on the motion: {topic}\n"
               "Write plain prose, no markdown, at most 60 words. "
               "If your opponent has spoken, rebut their latest point first. "
               "Only state facts you are confident are accurate."),
    ("human", "Debate so far:\n{transcript}\n\nYour turn."),
])

class Verdict(BaseModel):
    factual_errors: list[str] = Field(
        description="Claims that are factually wrong, each with the correction")
    unsupported_claims: list[str] = Field(
        description="Claims presented as fact with no specifics or evidence")
    reasoning: str = Field(description="Weigh both sides. Do not credit any claim "
        "listed above as an error or unsupported. At most 80 words")
    winner: Literal["FOR", "AGAINST"]

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial debate judge with strong domain knowledge. "
               "Judge on factual accuracy and quality of reasoning, not on tone or "
               "confidence. Each argument carries style tags (0 to 1) from a separate "
               "classifier. They describe style only, not truth; treat them as hints."),
    ("human", "Motion: {topic}\n\nTranscript:\n{transcript}"),
])

class DebateState(TypedDict):
    topic: str
    round: int
    max_rounds: int
    transcript: Annotated[list[dict], operator.add]
    verdict: dict

def speak(side, llm, state):
    chain = debate_prompt | llm | StrOutputParser()
    text = chain.invoke({"side": side, "topic": state["topic"],
                         "transcript": fmt(state["transcript"])}).strip()
    return {"side": side, "round": state["round"], "text": text, "tags": tag(text)}

def pro_node(state):
    return {"transcript": [speak("FOR", pro_llm, state)]}

def con_node(state):
    return {"transcript": [speak("AGAINST", con_llm, state)],
            "round": state["round"] + 1}

def judge_node(state):
    chain = judge_prompt | judge_llm.with_structured_output(Verdict)
    v = chain.invoke({"topic": state["topic"],
                      "transcript": fmt(state["transcript"], with_tags=True)})
    return {"verdict": v.model_dump()}

def next_step(state):
    return "pro" if state["round"] <= state["max_rounds"] else "judge"

graph = StateGraph(DebateState)
graph.add_node("pro", pro_node)
graph.add_node("con", con_node)
graph.add_node("judge", judge_node)
graph.add_edge(START, "pro")
graph.add_edge("pro", "con")
graph.add_conditional_edges("con", next_step, ["pro", "judge"])
graph.add_edge("judge", END)
app = graph.compile()

inputs = {"topic": "Airbus fly-by-wire design is safer than Boeing's.",
          "round": 1, "max_rounds": 3, "transcript": []}

for update in app.stream(inputs, stream_mode="updates"):
    for node, data in update.items():
        if node in ("pro", "con"):
            e = data["transcript"][-1]
            print(f"[{e['side']} r{e['round']}] {e['text']}")
            print(f"   tags: {e['tags']}\n")
        elif node == "judge":
            v = data["verdict"]
            print("=" * 60)
            print(f"WINNER: {v['winner']}")
            print(f"REASONING: {v['reasoning']}")
            print("FACTUAL ERRORS:")
            for x in v["factual_errors"]:
                print(f"  - {x}")
            print("UNSUPPORTED CLAIMS:")
            for x in v["unsupported_claims"]:
                print(f"  - {x}")
