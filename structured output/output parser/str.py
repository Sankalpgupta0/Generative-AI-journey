# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_completion_tokens=2,
)

# 1st prompt template
temp1 = PromptTemplate(
    template="write a detailed Report on {topic}",
    input_variables=["topic"],
)

# 2nd prompt template
temp2 = PromptTemplate(
    template="write a 5 line summary on the Following text. \n {text}",
    input_variables=["text"],
)

# propmt1 = temp1.invoke({
#     "topic" : "Artificial Intelligence"
# })

# result1 = model.invoke(propmt1)

# propmt2 = temp2.invoke({
#     "text" : result1.content
# })

# result2 = model.invoke(propmt2)
# print(result2.content)

parser = StrOutputParser()

chain = temp1 | model | parser | temp2 | model | parser

result = chain.invoke({
    "topic" : "Artificial Intelligence"
})
print(result)