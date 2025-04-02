from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0,
    max_completion_tokens=20,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is langchain?"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)