# Extract specific pages from a PDF and save them as a new PDF in the same directory
from pathlib import Path
from pypdf import PdfReader, PdfWriter

def extract_pages(pdf_path, output_path, page_numbers):
    """
    Extract specified page numbers from a PDF and save as a new PDF file.
    """
    try:
        # Ensure the output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open the original PDF file
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        # Extract specified pages
        for page_number in page_numbers:
            if 1 <= page_number <= len(reader.pages):
                writer.add_page(reader.pages[page_number - 1])
            else:
                print(f"Warning: Page number {page_number} is out of range. The PDF has {len(reader.pages)} pages.")

        # Save the new PDF file
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

        print(f"Successfully extracted pages {page_numbers} to {output_path}")

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

if __name__ == "__main__":
    pdf_path = "90-Data/complex-pdf/uber_10q_march_2022.pdf"
    output_path = "90-Data/complex-pdf/uber_10q_march_2022_page1-3.pdf"
    page_numbers = [26, 27, 28]  # Specify the page numbers to extract
    extract_pages(pdf_path, output_path, page_numbers)
