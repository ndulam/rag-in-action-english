# Install dependencies: pip install pymilvus
from pymilvus import MilvusClient, DataType

# ——————————————
# 0. Connect to Milvus
# ——————————————
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
print("✓ Connected to Milvus interface")

# ——————————————
# 1. Create a basic Schema
# ——————————————
schema = MilvusClient.create_schema()
print("✓ Created empty Schema")

# ——————————————
# 2. Add Primary Key field (Primary Field)
# ——————————————
# 2.1 INT64 type primary key (manually specified ID)
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,  # Set as primary key
    auto_id=False     # Do not auto-generate ID
)

# 2.2 VARCHAR type primary key (auto-generated ID)
# schema.add_field(
#     field_name="doc_id",
#     datatype=DataType.VARCHAR,
#     is_primary=True,  # Set as primary key
#     auto_id=True,     # Auto-generate ID
#     max_length=100    # VARCHAR type requires max length specification
# )
print("✓ Added primary key field")

# ——————————————
# 3. Add Vector fields (Vector Field)
# ——————————————
# 3.1 Dense Vector (float vector)
schema.add_field(
    field_name="text_vector",
    datatype=DataType.FLOAT_VECTOR,  # 32-bit float vector
    dim=768                          # Vector dimensions
)

# 3.2 Binary Vector (binary vector)
schema.add_field(
    field_name="image_vector",
    datatype=DataType.BINARY_VECTOR,  # Binary vector
    dim=256                           # Dimensions must be a multiple of 8
)
print("✓ Added vector fields")

# ——————————————
# 4. Add Scalar fields (Scalar Field)
# ——————————————
# 4.1 String field
schema.add_field(
    field_name="title",
    datatype=DataType.VARCHAR,
    max_length=200,
    # Can be null and has a default value
    is_nullable=True,
    default_value="untitled"
)

# 4.2 Numeric field
schema.add_field(
    field_name="age",
    datatype=DataType.INT32,
    is_nullable=False  # Cannot be null
)

# 4.3 Boolean field
schema.add_field(
    field_name="is_active",
    datatype=DataType.BOOL,
    default_value=True  # Default value is True
)

# 4.4 JSON field
schema.add_field(
    field_name="metadata",
    datatype=DataType.JSON
)

# 4.5 Array field
schema.add_field(
    field_name="tags",
    datatype=DataType.ARRAY,
    element_type=DataType.VARCHAR,  # Array element type
    max_capacity=10,                # Maximum array capacity
    max_length=50                   # Maximum length per element
)
print("✓ Added scalar fields")

# ——————————————
# 5. Add Dynamic fields (Dynamic Field)
# ——————————————
# schema.add_field(
#     field_name="dynamic_field",
#     datatype=DataType.VARCHAR,
#     is_dynamic=True,    # Set as dynamic field
#     max_length=500
# )
print("✓ Added dynamic fields")

# ——————————————
# 6. Create Collection using the Schema
# ——————————————
collection_name = "document_store10"
client.create_collection(
    collection_name=collection_name,
    schema=schema
)
print(f"✓ Created collection {collection_name}")

# ——————————————
# 7. Modify Collection fields
# ——————————————
# Add new field
# client.alter_collection_field(
#     collection_name=collection_name,
#     field_name="tags",
#     field_params={
#         "max_capacity": 64
#     }
# )
# print("✓ Added new field")

# ——————————————
# 8. View Collection details
# ——————————————
info = client.describe_collection(collection_name=collection_name)
print("Collection details:", info)

# ——————————————
# 9. Cleanup
# ——————————————
client.drop_collection(collection_name=collection_name)
print("✓ Deleted test collection")
