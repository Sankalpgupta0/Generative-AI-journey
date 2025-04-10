from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_completion_tokens=2,
)

parser = JsonOutputParser()

temp = PromptTemplate(
    template="Give me five random people with their name, age, and city.\n{format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()},
)

# prompt = temp.format()

# result = model.invoke(prompt)

chain = temp | model | parser
result = chain.invoke({})
print(result)
