from langchain_community.document_loaders import UnstructuredImageLoader
image_path = "90-Data/black-myth-wukong/black-myth-wukong-english.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)
