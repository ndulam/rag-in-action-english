import pymupdf
# Open the PDF file
doc = pymupdf.open("90-Data/black-myth-wukong/black-myth-wukong.pdf")
text = [page.get_text() for page in doc]
print(text)

# Example: Using the basic features of PyMuPDF
print("=== PyMuPDF Basic Information Extraction ===")
print(f"Number of pages: {len(doc)}")
print(f"Document title: {doc.metadata['title']}")
print(f"Document author: {doc.metadata['author']}")
print(f"Document metadata: {doc.metadata}")  # Provides more metadata than Unstructured

# Iterate over each page
for page_num, page in enumerate(doc):
    # Extract text
    text = page.get_text()
    print(f"\n--- Page {page_num + 1} ---")
    print("Text content:", text[:200])  # Show first 200 characters

    # Extract images
    images = page.get_images()
    print(f"Number of images: {len(images)}")

    # Get page links
    links = page.get_links()
    print(f"Number of links: {len(links)}")

    # Get page dimensions
    width, height = page.rect.width, page.rect.height
    print(f"Page dimensions: {width} x {height}")

doc.close()

# PyMuPDF (fitz) vs Unstructured comparison:
# Advantages:
# 1. Faster processing speed
# 2. Finer-grained PDF control
# 3. Access to more metadata and document structure information
# 4. Lower memory usage
# 5. No dependency on external tools

# Disadvantages:
# 1. Lower intelligence in text extraction
# 2. No automatic document structure understanding
# 3. Manual layout analysis required
