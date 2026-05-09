import pandas as pd
import os

# Define list of CSV files to process
csv_files = [
    'billionaires_table_2.csv',
    'billionaires_table_3.csv',
    'billionaires_table_4.csv',
    'billionaires_table_5.csv',
    'billionaires_table_6.csv'
]

# Create an ExcelWriter object
with pd.ExcelWriter('billionaires_merged.xlsx', engine='openpyxl') as writer:
    # Iterate over each CSV file
    for csv_file in csv_files:
        # Read CSV file
        df = pd.read_csv(csv_file)

        # Get sheet name (strip .csv extension)
        sheet_name = os.path.splitext(csv_file)[0]

        # Write data to the corresponding sheet in Excel
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("CSV files successfully merged into billionaires_merged.xlsx") 