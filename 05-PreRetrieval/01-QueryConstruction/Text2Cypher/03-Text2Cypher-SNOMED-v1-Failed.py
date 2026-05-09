# Prepare Neo4j database connection
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

# Neo4j connection configuration
uri = "bolt://localhost:7687"  # Default Neo4j Bolt port
username = "neo4j"
password = os.getenv("NEO4J_PASSWORD")  # Get password from environment variable

# Initialize Neo4j driver
driver = GraphDatabase.driver(uri, auth=(username, password))

# Prepare SNOMED CT schema description
schema_description = """
You are accessing a SNOMED CT graph database containing the following main nodes and relationships:

Node types:
1. Concept (Concept node)
   - conceptId: Unique concept identifier
   - fullySpecifiedName: Fully specified concept name
   - preferredTerm: Preferred term
   - active: Whether active
   - effectiveTime: Effective time
   - moduleId: Module ID

2. Description (Description node)
   - descriptionId: Unique description identifier
   - term: Term text
   - typeId: Description type ID
   - languageCode: Language code
   - active: Whether active

3. Relationship (Relationship node)
   - relationshipId: Unique relationship identifier
   - typeId: Relationship type ID
   - active: Whether active

Relationship types:
1. IS_A: Represents hierarchical relationships between concepts
2. HAS_DESCRIPTION: Relationship between a concept and its descriptions
3. HAS_RELATIONSHIP: Other relationships between concepts
"""

# Initialize DeepSeek client
from openai import OpenAI
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Set the query
user_query = "Find all concepts related to 'Diabetes' and their descriptions"

# Prepare the prompt for generating Cypher
prompt = f"""
The following is the schema description of a SNOMED CT graph database:
{schema_description}
The user's natural language question is:
"{user_query}"
Please note:
1. Use the MATCH clause to match nodes and relationships
2. Use the WHERE clause for filtering conditions
3. Use the RETURN clause to specify return results
4. Return only the Cypher query statement without any other explanations, comments, or format markers (such as ```cypher)
"""

# Call the LLM to generate the Cypher statement
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a Cypher query expert. Return only the Cypher query statement without any Markdown formatting or other explanations."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

# Clean the Cypher statement, remove possible Markdown markers
cypher = response.choices[0].message.content.strip()
cypher = cypher.replace('```cypher', '').replace('```', '').strip()
print(f"\nGenerated Cypher query:\n{cypher}")

# Execute the Cypher query and retrieve results
def run_query(tx, query):
    result = tx.run(query)
    return [record for record in result]

with driver.session() as session:
    results = session.execute_read(run_query, cypher)
    print(f"Query results: {results}")

# Generate natural language description
if results:
    nl_prompt = f"""
Query results:
{results}
Please convert these data into a natural language description that is easy to understand.
The original question is: {user_query}

Requirements:
1. Use plain, easy-to-understand language
2. Include all queried data information
3. Explain any technical terms where appropriate
"""
    response_nl = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a medical information expert responsible for converting SNOMED CT query results into easy-to-understand natural language descriptions."},
            {"role": "user", "content": nl_prompt}
        ],
        temperature=0.7
    )
    description = response_nl.choices[0].message.content.strip()
    print(f"Natural language description:\n{description}")
else:
    print("No relevant data found.")

# Close database connection
driver.close()
