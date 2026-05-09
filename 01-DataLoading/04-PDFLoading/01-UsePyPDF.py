from langchain_community.document_loaders import PyPDFLoader
file_path = "90-Data/black-myth-wukong/black-myth-wukong.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
print(f"Loaded {len(pages)} pages from the PDF document")
for page in pages:
    print(page.page_content)
