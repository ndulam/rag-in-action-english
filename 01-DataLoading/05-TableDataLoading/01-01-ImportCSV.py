from langchain_community.document_loaders import CSVLoader
# # Part 1: Basic CSV file loading and printing records
file_path = "90-Data/black-myth-wukong/black-myth-wukong.csv"
# loader = CSVLoader(file_path=file_path)
# data = loader.load()
# print("Example 1: Basic CSV file loading and printing the first two records")
# for record in data[:2]:
#     print(record)
# print("-" * 80)

# Part 2: Skip the CSV header row and use custom column names
# loader = CSVLoader(
#     file_path=file_path,
#     csv_args={
#         "delimiter": ",",
#         "quotechar": '"',
#         "fieldnames": ["Type", "Name", "Description", "Level"],
#     },
# )
# data = loader.load()

# print("Example 2: Skip header row and use custom column names")
# for record in data[:2]:
#     print(record)
# print("-" * 80)


# # Part 3: Specify the "Name" column as source_column
# loader = CSVLoader(file_path=file_path, source_column="Name")
# data = loader.load()

# print("Example 3: Use the 'Name' column as the primary content source")
# for record in data[:2]:
#     print(record)
# print("-" * 80)


# Part 4: Use UnstructuredCSVLoader to load the CSV file
from langchain_community.document_loaders import UnstructuredCSVLoader
loader = UnstructuredCSVLoader(file_path=file_path)
data = loader.load()
print("Example 4: Use UnstructuredCSVLoader to load the file")
print(data)
print("-" * 80)
