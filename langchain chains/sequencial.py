from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_output_tokens=512,
)

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Make a detailed report on {topic}. upto 2000 words.",
)

prompt2 = PromptTemplate(
    input_variables=["report"],
    template="get five best points from the report {report}",
)

output_parser = StrOutputParser()

chain = prompt1 | llm | output_parser | prompt2 | llm | output_parser
result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)