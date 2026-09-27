from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatBedrockConverse(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    max_tokens=200,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a debater arguing the {side} side. "
               "Write plain prose only: no headings, no markdown. "
               "Maximum {max_sentences} sentences."),
    ("human", "Motion: {topic}"),
])

chain = prompt | llm | StrOutputParser()

for side in ["FOR", "AGAINST"]:
    print(f"--- {side} ---")
    print(chain.invoke({
        "side": side,
        "topic": "Airbus fly-by-wire design is safer than Boeing's.",
        "max_sentences": 2,
    }))
    print()
