from calendar import c
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda

import pydantic
from regex import R

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback")
    feedback : str = Field(description="Feedback text")
    
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

model = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-1.5-pro",
    max_output_tokens=512,
)

str_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Classify the sentiment of this {Feedback}?. Into positive and negative only \n {format_instruction}",
    input_variables=["Feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()},
)

classifer_chain = prompt1 | model | pydantic_parser

prompt2 = PromptTemplate(
    template="write an Appropriate feedback to this positive this {review}",
    input_variables=["review"],
)

prompt3 = PromptTemplate(
    template="write an Appropriate feedback to this Negative this {review}",
    input_variables=["review"],
)

# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default_chain
# )

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | str_parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | str_parser),
    RunnableLambda(lambda x: "No sentiment detected")
)

chain = classifer_chain | branch_chain

result = chain.invoke(
    {
        "Feedback": "The phone is great, but the battery life is terrible.",
    }
)

print(result)