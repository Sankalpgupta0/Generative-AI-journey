from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Mumbai is the capital of Maharashtra",
    "Chennai is the capital of Tamil Nadu",
    "Bangalore is the capital of Karnataka",
]

vector = embeddings.embed_documents(documents)

print(vector)