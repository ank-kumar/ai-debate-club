from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
    max_tokens=200,
)

response = llm.invoke(
    "In two sentences, argue that Airbus fly-by-wire design is safer than Boeing's."
)

print("TYPE:", type(response).__name__)
print("CONTENT:", response.content)
print("TOKENS:", response.usage_metadata)
