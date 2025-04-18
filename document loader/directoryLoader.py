from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='data',
    glob='*.pdf',
    loader_cls=PyPDFLoader
) 
# docs = loader.load()
# print(docs[18].page_content)

docs = loader.lazy_load()

for x in docs:
    print(x.page_content)
