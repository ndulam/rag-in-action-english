# Prepare database connection
import sqlite3
conn = sqlite3.connect('data/tourism.db')
cursor = conn.cursor()

# Prepare schema description
schema_description = """
You are accessing a database containing two tables:
1. scenic_spots (Scenic spot information table)
   - scenic_id (INT): Primary key, unique scenic spot identifier
   - scenic_name (VARCHAR): Scenic spot name
   - city (VARCHAR): City where it is located
   - level (VARCHAR): Scenic spot grade
   - monthly_visitors (INT): Number of visitors that month
2. city_info (City information table)
   - city_id (INT): Primary key, unique city identifier
   - city_name (VARCHAR): City name
   - annual_tourism_income (INT): Annual tourism revenue (unit: yuan)
   - famous_dish (VARCHAR): Local famous dish/specialty food
"""

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI()

# Set the query
user_query = "Query the AAAAA-grade scenic spots in Taiyuan City and their monthly visitor counts"

# Prepare the prompt for generating SQL
prompt = f"""
The following is the database schema description:
{schema_description}
The user's natural language question is:
"{user_query}"
Please note:
1. The city field in the scenic_spots table stores city names, corresponding to the city_name in the city_info table
2. The join between the two tables should use city_name and city for matching
3. Return only the SQL query statement without any other explanations, comments, or format markers (such as ```sql)
"""
# Call the LLM to generate the SQL statement
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a SQL expert. Return only the SQL query statement without any Markdown formatting or other explanations."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

# Clean the SQL statement, remove possible Markdown markers
sql = response.choices[0].message.content.strip()
sql = sql.replace('```sql', '').replace('```', '').strip()
print(f"\nGenerated SQL query:\n{sql}")

# Execute SQL and retrieve results
cursor.execute(sql)
results = cursor.fetchall()
print(f"Query results: {results}")
conn.close()

# Generate natural language description
if results:
    # Get column names
    column_names = [description[0] for description in cursor.description]
    # Convert results to list of dicts
    results_with_columns = [dict(zip(column_names, row)) for row in results]
    nl_prompt = f"""
Query results:
{results_with_columns}
Please convert these data into a natural language description that is easy to understand.
The original question is: {user_query}

Requirements:
1. Use plain, easy-to-understand language
2. Include all queried data information
3. Express numbers in words where appropriate
"""
    response_nl = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data analyst responsible for converting query results into easy-to-understand natural language descriptions."},
            {"role": "user", "content": nl_prompt}
        ],
        temperature=0.7
    )
    description = response_nl.choices[0].message.content.strip()
    print(f"Natural language description:\n{description}")
else:
    print("No relevant data found.")
# Close database connection
conn.close()
