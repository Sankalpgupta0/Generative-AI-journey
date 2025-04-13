from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the Google Generative AI chat model
llm = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-1.5-pro",
    max_output_tokens=256,
)

# Define the prompt template
prompt = PromptTemplate(
    template="Generate a five facts about {topic}",
    input_variables=["topic"],
)

# Define the output parser
parser = StrOutputParser()

# Combine the prompt and the model into a chain
chain = prompt | llm | parser

result = chain.invoke({"topic":"black holes"})

print(result)

chain.get_graph().print_ascii()