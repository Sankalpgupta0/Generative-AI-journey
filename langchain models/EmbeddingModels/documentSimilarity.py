from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
documents_vector = embeddings.embed_documents(documents)

query = "Who is prime minister of India?"
query_vector = embeddings.embed_query(query)

# both query_vector and documents_vector are 2D arrays
# Calculate cosine similarity
similarity = cosine_similarity([query_vector], documents_vector)[0]

# Get the index of the most similar document
index, score = sorted(list(enumerate(similarity)), key=lambda x: x[1])[-1]
print(f"Most similar document is: {documents[index]} with score: {score}")