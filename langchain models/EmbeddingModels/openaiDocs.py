from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Mumbai is the capital of Maharashtra",
    "Chennai is the capital of Tamil Nadu",
    "Bangalore is the capital of Karnataka",
]
embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    dimensions=32
)

result = embedding.embed_documents(documents)
print(str(result))