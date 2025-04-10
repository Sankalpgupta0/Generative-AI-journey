from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_completion_tokens=2,
)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

class People(BaseModel):
    people: List[Person]
    
parser = PydanticOutputParser(pydantic_object=People)

template = PromptTemplate(
    template='Generate 5 names, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'India'})

print(final_result)