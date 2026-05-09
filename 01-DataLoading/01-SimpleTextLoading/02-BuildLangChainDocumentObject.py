from langchain_core.documents import Document
documents = [
    Document(
        page_content="Sun Wukong is the eldest disciple.",
        metadata={"source": "four-disciples.txt"},
    ),
    Document(
        page_content="Zhu Bajie is the second disciple.",
        metadata={"source": "four-disciples.txt "},
    ),
]
print(documents)
