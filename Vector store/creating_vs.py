from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # Use langchain_core instead of langchain.schema

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Create a list of movie documents
movies = [
    Document(
        page_content="A sci-fi adventure through space and time with a father and daughter bond.",
        metadata={"title": "Interstellar", "director": "Christopher Nolan", "year": 2014},
    ),
    Document(
        page_content="A thief who steals corporate secrets through dream-sharing technology is given a task to plant an idea.",
        metadata={"title": "Inception", "director": "Christopher Nolan", "year": 2010},
    ),
    Document(
        page_content="A brilliant mathematician uncovers patterns in stock markets while battling mental illness.",
        metadata={"title": "A Beautiful Mind", "director": "Ron Howard", "year": 2001},
    ),
    Document(
        page_content="A Marvel superhero embarks on a multiversal journey to protect the fabric of reality.",
        metadata={"title": "Doctor Strange in the Multiverse of Madness", "director": "Sam Raimi", "year": 2022},
    ),
    Document(
        page_content="A gripping portrayal of the development of the atomic bomb during World War II.",
        metadata={"title": "Oppenheimer", "director": "Christopher Nolan", "year": 2023},
    ),
]

# Correct usage: use from_documents
vector_stores = Chroma.from_documents(
    documents=movies,
    embedding=embeddings,
    collection_name="movies",
    persist_directory="chroma_movies",
)
