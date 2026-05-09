from langchain_community.document_loaders import TextLoader
print("=== TextLoader Loading Result ===")
text_loader = TextLoader("90-Data/chronicles-of-gods/characters.json")
text_documents = text_loader.load()
print(text_documents)
