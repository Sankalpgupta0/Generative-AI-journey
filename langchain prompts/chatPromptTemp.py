from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You are a very knowledgeable {domain} expert."),
    ('human', "What is {topic}?"),
])

# this does not works
# chat_template = ChatPromptTemplate([
#     SystemMessage(content="You are a very knowledgeable {domain} expert."),
#     HumanMessage(content="What is {topic}?"),
# ])

prompt = chat_template.invoke({
        "domain":"Python",
        "topic":"langchain",
    }
)

print(prompt)