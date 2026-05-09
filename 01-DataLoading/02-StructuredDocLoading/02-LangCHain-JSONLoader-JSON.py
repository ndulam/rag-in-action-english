from langchain_community.document_loaders import JSONLoader
print("=== JSONLoader Loading Result ===")
print("1. Main character information:")
main_loader = JSONLoader(
    file_path="90-Data/chronicles-of-gods/characters.json",
    jq_schema='.mainCharacter | "Name: " + .name + ", Background: " + .backstory',
    text_content=True
)
main_char = main_loader.load()
print(main_char)
print("\n2. Supporting character information:")
support_loader = JSONLoader(
    file_path="90-Data/chronicles-of-gods/characters.json",
    jq_schema='.supportCharacters[] | "Name: " + .name + ", Background: " + .background',
    text_content=True
)
support_chars = support_loader.load()
print(support_chars)
