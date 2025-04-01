from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv() 

model = ChatAnthropic(
    model="claude-2",
    temperature=0,
    max_tokens=2000,
)

result = model.invoke("What is the capital of France?")
print(result)