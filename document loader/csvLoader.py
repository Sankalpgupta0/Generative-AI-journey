from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="./data/data.csv")

docs = loader.load()

# print(len(docs))
print(docs[0].page_content)