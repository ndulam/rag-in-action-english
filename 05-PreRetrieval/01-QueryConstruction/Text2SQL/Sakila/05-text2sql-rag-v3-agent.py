# text2sql_query.py
import os
import logging
import yaml
import openai
import re
from dotenv import load_dotenv
from pymilvus import MilvusClient
from pymilvus import model
from sqlalchemy import create_engine, text

# 1. Environment and logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # Load environment variables from .env

# 2. Initialize OpenAI API (using the latest Response API)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Recommended to use new Response API style
# e.g.: openai.chat.completions.create(...) instead of the old ChatCompletion.create

MODEL_NAME = os.getenv("OPENAI_MODEL", "o4-mini")

# 3. Embedding function initialization
def init_embedding():
    return model.dense.OpenAIEmbeddingFunction(
        model_name='text-embedding-3-large',
    )

# 4. Milvus client connection
MILVUS_DB = os.getenv("MILVUS_DB_PATH", "text2sql_milvus_sakila.db")
client = MilvusClient(MILVUS_DB)

# 5. Instantiate embedding function
embedding_fn = init_embedding()

# 6. Database connection (SAKILA)
DB_URL = os.getenv(
    "SAKILA_DB_URL",
    "mysql+pymysql://root:password@localhost:3306/sakila"
)
engine = create_engine(DB_URL)

# 7. Retrieval function
def retrieve(collection: str, query_emb: list, top_k: int = 3, fields: list = None):
    results = client.search(
        collection_name=collection,
        data=[query_emb],
        limit=top_k,
        output_fields=fields
    )
    logging.info(f"[Retrieval] {collection} search results: {results}")
    return results[0]  # Return the result list for the first query

# 8. SQL extraction function
def extract_sql(text: str) -> str:
    # Try to match SQL code block
    sql_blocks = re.findall(r'```sql\n(.*?)\n```', text, re.DOTALL)
    if sql_blocks:
        return sql_blocks[0].strip()

    # If no code block found, try to match SELECT statement
    select_match = re.search(r'SELECT.*?;', text, re.DOTALL)
    if select_match:
        return select_match.group(0).strip()

    # If neither found, return the raw text
    return text.strip()

# 9. Execute SQL and return results
def execute_sql(sql: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            cols = result.keys()
            rows = result.fetchall()
            return True, cols, rows
    except Exception as e:
        return False, None, str(e)

# 10. SQL generation function
def generate_sql(prompt: str, error_msg: str = None) -> str:
    if error_msg:
        prompt += f"\nThe previous SQL execution failed with error: {error_msg}\nPlease correct the SQL statement:"

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_sql = response.choices[0].message.content.strip()
    sql = extract_sql(raw_sql)
    logging.info(f"[Generation] Raw output: {raw_sql}")
    logging.info(f"[Generation] Extracted SQL: {sql}")
    return sql

# 11. Core pipeline: natural language -> SQL -> execute -> return
def text2sql(question: str, max_retries: int = 3):
    # 11.1 Embed the user question
    q_emb = embedding_fn([question])[0]
    logging.info(f"[Retrieval] Question embedding complete")

    # 11.2 RAG retrieval: DDL
    ddl_hits = retrieve("ddl_knowledge", q_emb.tolist(), top_k=3, fields=["ddl_text"])
    logging.info(f"[Retrieval] DDL retrieval results: {ddl_hits}")
    try:
        ddl_context = "\n".join(hit.get("ddl_text", "") for hit in ddl_hits)
    except Exception as e:
        logging.error(f"[Retrieval] DDL processing error: {e}")
        ddl_context = ""

    # 11.3 RAG retrieval: example pairs
    q2sql_hits = retrieve("q2sql_knowledge", q_emb.tolist(), top_k=3, fields=["question", "sql_text"])
    logging.info(f"[Retrieval] Q2SQL retrieval results: {q2sql_hits}")
    try:
        example_context = "\n".join(
            f"NL: \"{hit.get('question', '')}\"\nSQL: \"{hit.get('sql_text', '')}\""
            for hit in q2sql_hits
        )
    except Exception as e:
        logging.error(f"[Retrieval] Q2SQL processing error: {e}")
        example_context = ""

    # 11.4 RAG retrieval: field descriptions
    desc_hits = retrieve("dbdesc_knowledge", q_emb.tolist(), top_k=5, fields=["table_name", "column_name", "description"])
    logging.info(f"[Retrieval] Field description retrieval results: {desc_hits}")
    try:
        desc_context = "\n".join(
            f"{hit.get('table_name', '')}.{hit.get('column_name', '')}: {hit.get('description', '')}"
            for hit in desc_hits
        )
    except Exception as e:
        logging.error(f"[Retrieval] Field description processing error: {e}")
        desc_context = ""

    # 11.5 Assemble base prompt
    base_prompt = (
        f"### Schema Definitions:\n{ddl_context}\n"
        f"### Field Descriptions:\n{desc_context}\n"
        f"### Examples:\n{example_context}\n"
        f"### Query:\n\"{question}\"\n"
        "Return only the SQL statement, without any explanation or commentary."
    )

    # 11.6 Generate and execute SQL, retry up to max_retries times
    error_msg = None
    for attempt in range(max_retries):
        logging.info(f"[Execution] Attempt {attempt + 1}")

        # Generate SQL
        sql = generate_sql(base_prompt, error_msg)

        # Execute SQL
        success, cols, result = execute_sql(sql)

        if success:
            print("\nQuery Results:")
            print("Columns:", cols)
            for r in result:
                print(r)
            return

        error_msg = result
        logging.error(f"[Execution] Attempt {attempt + 1} failed: {error_msg}")

    print(f"Execution failed after reaching the maximum number of retries ({max_retries}).")
    print("Last error:", error_msg)

# 12. Program entry point
if __name__ == "__main__":
    user_q = input("Please enter your natural language query: ")
    text2sql(user_q)
