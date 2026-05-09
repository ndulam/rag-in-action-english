# generate_ddl_yaml.py
import os
import yaml
import pymysql
from dotenv import load_dotenv

# 1. Load database configuration from .env
load_dotenv()  

host = os.getenv("MYSQL_HOST")
port = int(os.getenv("MYSQL_PORT", "3306"))
user = "root"
password = "password"
db_name = "sakila"

# 2. Connect to MySQL``
conn = pymysql.connect(
    host=host, port=port, user=user, password=password,
    database=db_name, cursorclass=pymysql.cursors.Cursor
)  

ddl_map = {}
try:
    with conn.cursor() as cursor:
        # 3. Retrieve all table names
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s;", (db_name,)
        )  
        tables = [row[0] for row in cursor.fetchall()]

        # 4. Iterate over table list and run SHOW CREATE TABLE
        for tbl in tables:
            cursor.execute(f"SHOW CREATE TABLE `{db_name}`.`{tbl}`;")
            result = cursor.fetchone()
            # result[0]=table name, result[1]=full DDL
            ddl_map[tbl] = result[1]  

finally:
    conn.close()

# 5. Write to YAML file
with open("90-Data/sakila/ddl_statements.yaml", "w") as f:
    yaml.safe_dump(ddl_map, f, sort_keys=True, allow_unicode=True)
print("✅ ddl_statements.yaml generated, tables included:", list(ddl_map.keys()))
