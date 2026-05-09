import fitz  # PyMuPDF library for handling PDF files
import matplotlib.patches as patches  # For drawing polygons on images
import matplotlib.pyplot as plt  # Matplotlib library for plotting
from PIL import Image  # For image processing

def render_pdf_page(file_path, doc_list, page_number):
    """
    Render a specified PDF page and draw paragraph classification boxes.

    Parameters:
    - file_path: str, path to the PDF file.
    - doc_list: list, a list of documents containing paragraph information; each element is a dict with paragraph metadata.
    - page_number: int, the page number to render (starting from 1).
    """
    # Open the PDF file and load the specified page
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    segments = [doc.metadata for doc in doc_list if doc.metadata.get("page_number") == page_number]

    # Render the PDF page as a bitmap image
    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Create the plotting environment
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pil_image)

    # Define category-to-color mapping
    category_to_color = {"Title": "orchid", "Image": "forestgreen", "Table": "tomato"}
    categories = set()

    # Draw paragraph annotation boxes
    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]
        scaled_points = [
            (x * pix.width / layout_width, y * pix.height / layout_height) for x, y in points
        ]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(scaled_points, linewidth=1, edgecolor=box_color, facecolor="none")
        ax.add_patch(rect)

    # Add legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category, color in category_to_color.items():
        if category in categories:
            legend_handles.append(patches.Patch(color=color, label=category))
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
