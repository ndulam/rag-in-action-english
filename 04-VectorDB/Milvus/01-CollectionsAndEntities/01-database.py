# Install dependencies: pip install pymilvus
# pip show pymilvus  # Check current SDK version
'''
# Install Milvus server

wget https://github.com/milvus-io/milvus/releases/download/v2.5.10/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

'''

from pymilvus import MilvusClient, exceptions

# ——————————————
# 1. Connect to Milvus Standalone
# ——————————————
# uri: protocol + address + port, default is http://localhost:19530
# token: "username:password", default root:Milvus
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# ——————————————
# 2. Create database my_database_1 (no additional properties)
# ——————————————
try:
    client.create_database(db_name="my_database_1")
    print("✓ my_database_1 created successfully")
except exceptions.AlreadyExistError:
    print("ℹ my_database_1 already exists")

# ——————————————
# 3. Create database my_database_2 (set replica count to 3)
# ——————————————
client.create_database(
    db_name="my_database_2",
    properties={"database.replica.number": 3}
)
print("✓ my_database_2 created successfully, replica count=3")

# ——————————————
# 4. List all databases
# ——————————————
db_list = client.list_databases()
print("All current databases:", db_list)

# ——————————————
# 5. View details of the default database
# ——————————————
default_info = client.describe_database(db_name="default")
print("Default database details:", default_info)

# ——————————————
# 6. Modify my_database_1 property: limit max collections to 10
# ——————————————
client.alter_database_properties(
    db_name="my_database_1",
    properties={"database.max.collections": 10}
)
print("✓ Set max collections limit to 10 for my_database_1")

# ——————————————
# 7. Remove the max.collections limit from my_database_1
# ——————————————
client.drop_database_properties(
    db_name="my_database_1",
    property_keys=["database.max.collections"]
)
print("✓ Removed max collections limit from my_database_1")

# ——————————————
# 8. Switch to my_database_2 (all subsequent operations apply to this database)
# ——————————————
client.use_database(db_name="my_database_2")
print("✓ Switched current database to my_database_2")

# ——————————————
# 9. Drop database my_database_2
#    (Note: if the database contains Collections, first use client.drop_collection() to clear them)
# ——————————————
client.drop_database(db_name="my_database_2")
print("✓ my_database_2 deleted")

# ——————————————
# 10. Drop database my_database_1
# ——————————————
client.drop_database(db_name="my_database_1")
print("✓ my_database_1 deleted")
