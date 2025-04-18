from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("data.txt")
parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_completion_tokens=2,
)

prompt = PromptTemplate(
    template="write a professional summary about: {input}",
    input_variables=["input"],
)

docs = loader.load()

chain = prompt | model | parser # RunnableSequence(prompt, llm, parser])

print(chain.invoke({'input' : docs[0].page_content}))
# print(type(docs))
# print(len(docs))