# Install dependencies: pip install pymilvus
from pymilvus import MilvusClient

# ——————————————
# 0. Connect to Milvus
# ——————————————
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
print("✓ Connected to Milvus interface")

# ——————————————
# 1. Create Collection (quick setup mode)
# ——————————————
# Check and drop existing collection
collection_name = "quick_setup"
if collection_name in client.list_collections():
    client.drop_collection(collection_name=collection_name)
    print(f"✓ Dropped existing collection {collection_name}")

# Create new collection
client.create_collection(
    collection_name=collection_name,
    dimension=5
)
print(f"✓ {collection_name} created")

# ——————————————
# 2. List all Collections
# ——————————————
cols = client.list_collections()
print("All current collections:", cols)

# ——————————————
# 3. View Collection details
# ——————————————
info = client.describe_collection(collection_name=collection_name)
print(f"{collection_name} details:", info)

# ——————————————
# 4. Rename Collection
# ——————————————
new_collection_name = "quick_renamed"
if new_collection_name in client.list_collections():
    client.drop_collection(collection_name=new_collection_name)
    print(f"✓ Dropped existing collection {new_collection_name}")

client.rename_collection(
    old_name=collection_name,
    new_name=new_collection_name
)
print(f"✓ {collection_name} renamed to {new_collection_name}")

# ——————————————
# 5. Modify Collection properties (set TTL to 60 seconds)
# ——————————————
client.alter_collection_properties(
    collection_name=new_collection_name,
    properties={"collection.ttl.seconds": 60}
)
print(f"✓ Set TTL=60s for {new_collection_name}")

# ——————————————
# 6. Drop Collection properties (TTL)
# ——————————————
client.drop_collection_properties(
    collection_name=new_collection_name,
    property_keys=["collection.ttl.seconds"]
)
print(f"✓ Dropped TTL property from {new_collection_name}")

# ——————————————
# 7. Load & check load state
# ——————————————
client.load_collection(collection_name=new_collection_name)
state = client.get_load_state(collection_name=new_collection_name)
print("Load state:", state)

# ——————————————
# 8. Release & check release state
# ——————————————
client.release_collection(collection_name=new_collection_name)
state = client.get_load_state(collection_name=new_collection_name)
print("State after release:", state)

# ——————————————
# 9. Manage Partitions
# ——————————————
# 9.1 List Partitions (only "_default" by default)
parts = client.list_partitions(collection_name=new_collection_name)
print("Partition list:", parts)

# 9.2 Create new Partition
client.create_partition(
    collection_name=new_collection_name,
    partition_name="partA"
)
print("✓ Created partition partA")
print("Updated partition list:", client.list_partitions(new_collection_name))

# 9.3 Check if Partition exists
exists = client.has_partition(
    collection_name=new_collection_name,
    partition_name="partA"
)
print("partA exists?", exists)

# 9.4 Load & release specified Partition
client.load_partitions(
    collection_name=new_collection_name,
    partition_names=["partA"]
)
print("partA load state:", client.get_load_state(new_collection_name, partition_name="partA"))

client.release_partitions(
    collection_name=new_collection_name,
    partition_names=["partA"]
)
print("partA state after release:", client.get_load_state(new_collection_name, partition_name="partA"))

# 9.5 Drop Partition (must release first)
client.drop_partition(
    collection_name=new_collection_name,
    partition_name="partA"
)
print("✓ Dropped partition partA")
print("Final partition list:", client.list_partitions(new_collection_name))

# ——————————————
# 10. Manage Aliases
# ——————————————
# 10.1 Create Aliases
client.create_alias(collection_name=new_collection_name, alias="alias3")
client.create_alias(collection_name=new_collection_name, alias="alias4")
print("✓ Created alias3, alias4")

# 10.2 List Aliases
aliases = client.list_aliases(collection_name=new_collection_name)
print("Current aliases:", aliases)

# 10.3 View Alias details
desc = client.describe_alias(alias="alias3")
print("alias3 details:", desc)

# 10.4 Reassign Alias
client.alter_alias(collection_name=new_collection_name, alias="alias4")
print("✓ Reassigned alias4 to quick_renamed")

# 10.5 Drop Alias
client.drop_alias(alias="alias3")
print("✓ Dropped alias3")
print("Remaining aliases:", client.list_aliases(new_collection_name))

# ——————————————
# 11. Drop Collection
# ——————————————
client.drop_collection(collection_name=new_collection_name)
print(f"✓ Collection {new_collection_name} deleted")
