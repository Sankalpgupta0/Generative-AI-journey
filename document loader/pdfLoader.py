from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("resume.pdf").load()

print(docs[0].page_content)
# print(docs[0].metadata)