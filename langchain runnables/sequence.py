from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from regex import P

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_output_tokens=1024,
)

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain1 = RunnableSequence(prompt1, llm, parser)

prompt2 = PromptTemplate(
    template="explain this joke : {joke}",
    input_variables=["joke"],
)

chain2 = RunnableSequence(prompt2, llm, parser)

chain = RunnableSequence(chain1, chain2)

print(chain.invoke({"topic": "the sea"}))