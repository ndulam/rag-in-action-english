from sqlalchemy import create_engine, text
import pandas as pd

# Make sure to use pymysql as the driver
engine = create_engine("mysql+pymysql://newuser:password@localhost:3306/example_db")

# Test the connection
try:
    with engine.connect() as connection:
        # Wrap the SQL statement with the text() function
        result = connection.execute(text("SELECT * FROM game_scenes"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        print("Query result:")
        print(df)
except Exception as e:
    print("Database connection failed:", e)



