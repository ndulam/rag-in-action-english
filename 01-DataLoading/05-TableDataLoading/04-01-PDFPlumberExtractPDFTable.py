import pdfplumber
import pandas as pd
import time

# Record start time
start_time = time.time()

# Open the PDF file
pdf = pdfplumber.open("90-Data/complex-pdf/billionaires_page-1-5.pdf")

# Iterate over each page
for page in pdf.pages:
    # Extract tables
    tables = page.extract_tables()

    # Check if any tables were found
    if tables:
        print(f"Found {len(tables)} table(s) on page {page.page_number}")

        # Iterate over all tables on this page
        for i, table in enumerate(tables):
            print(f"\nProcessing table {i+1}:")

            # Convert the table to a DataFrame
            df = pd.DataFrame(table)

            # If the first row contains column names, set it as the header
            if len(df) > 0:
                df.columns = df.iloc[0]
                df = df.iloc[1:]  # Remove the duplicate header row

            print(df)
            print("-" * 50)

# Close the PDF
pdf.close()

# Record end time and calculate total elapsed time
end_time = time.time()
print(f"\nTotal PDF table extraction time: {end_time - start_time:.2f} seconds")
