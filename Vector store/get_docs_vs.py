from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Load existing vector store
vector_stores = Chroma(
    embedding_function=embeddings,
    collection_name="movies",
    persist_directory="chroma_movies",
)

# docs = vector_stores.similarity_search(
#     query='A computer hacker learns the true nature of reality and his role in the war against its controllers.',
#     k=2,
# )

docs = vector_stores.similarity_search_with_score(
    query='A computer hacker learns the true nature of reality and his role in the war against its controllers.',
    k=2,
)

# Print the documents
for doc in docs:
    print(doc)
