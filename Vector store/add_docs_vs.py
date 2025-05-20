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

# New documents to add
movie6 = Document(
    page_content="A young lion prince flees his kingdom only to learn the true meaning of responsibility and bravery.",
    metadata={"title": "The Lion King", "director": "Roger Allers & Rob Minkoff", "year": 1994},
)

movie7 = Document(
    page_content="A computer hacker learns the true nature of reality and his role in the war against its controllers.",
    metadata={"title": "The Matrix", "director": "Lana Wachowski & Lilly Wachowski", "year": 1999},
)

# Add to vector store
movies = [movie6, movie7]
vector_stores.add_documents(movies)
