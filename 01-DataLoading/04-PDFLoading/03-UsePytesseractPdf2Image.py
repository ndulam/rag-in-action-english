# For scanned image-type PDFs, pytesseract + pdf2image is recommended
# sudo apt-get install tesseract-ocr
# sudo apt-get install tesseract-ocr-chi-sim

import pdf2image
import pytesseract
import os

# Create the output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images and save
images = pdf2image.convert_from_path('90-Data/black-myth-wukong/black-myth-wukong.pdf')
for i, image in enumerate(images):
    image.save(f'{output_dir}/page_{i+1}.png')

# Use pytesseract to extract text
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang='chi_sim')
    print(f"Page {i+1} text:")
    print(text)
    print("\n")
