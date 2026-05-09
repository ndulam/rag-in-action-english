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

def get_database_schema():
    """Query database metadata"""
    with driver.session() as session:
        # Query node labels
        node_labels_query = """
        CALL db.labels() YIELD label
        RETURN label
        """
        node_labels = session.run(node_labels_query).data()

        # Query relationship types
        relationship_types_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        relationship_types = session.run(relationship_types_query).data()

        # Query properties for each label
        properties_by_label = {}
        for label in node_labels:
            properties_query = f"""
            MATCH (n:{label['label']})
            WITH n LIMIT 1
            RETURN keys(n) as properties
            """
            properties = session.run(properties_query).data()
            if properties:
                properties_by_label[label['label']] = properties[0]['properties']

        return {
            "node_labels": [label['label'] for label in node_labels],
            "relationship_types": [rel['relationshipType'] for rel in relationship_types],
            "properties_by_label": properties_by_label
        }

# Get database schema
schema_info = get_database_schema()
print("\nDatabase schema information:")
print("Node types:", schema_info["node_labels"])
print("Relationship types:", schema_info["relationship_types"])
print("\nNode properties:")
for label, properties in schema_info["properties_by_label"].items():
    print(f"{label}: {properties}")

# Prepare SNOMED CT schema description
schema_description = f"""
You are accessing a SNOMED CT graph database containing the following main nodes and relationships:

Node types:
{', '.join(schema_info["node_labels"])}

Relationship types:
{', '.join(schema_info["relationship_types"])}

Node properties:
"""
for label, properties in schema_info["properties_by_label"].items():
    schema_description += f"\n{label} node properties: {', '.join(properties)}"

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

Please generate a Cypher query statement, noting the following:
1. Relationship direction must be correct, for example:
   - ObjectConcept has Description, so it should be: (oc:ObjectConcept)-[:HAS_DESCRIPTION]->(d:Description)
   - Do NOT write: (d:Description)-[:HAS_DESCRIPTION]->(oc:ObjectConcept)
2. Use the MATCH clause to match nodes and relationships
3. Use the WHERE clause for filtering; it is recommended to use the toLower() function for case-insensitive matching
4. Use the RETURN clause to specify return results
5. Return only the Cypher query statement without any other explanations, comments, or format markers (such as ```cypher)
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
    print(f"\nQuery results: {results}")

# Close database connection
driver.close()
