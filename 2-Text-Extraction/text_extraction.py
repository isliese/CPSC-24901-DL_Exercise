# Joanna A Menghamal

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import zipfile
import io
import pytesseract
import pandas as pd
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust based on your `which tesseract` output


# Extracts text from a zip file and extracts information from receipt image files and saves to a CSV file

def extract_text_from_zip(zip_path):
    """
    Extracts text from images in a zip file.
    :param zip_path: Path to the zip file.
    :return: Dictionary with filenames as keys and extracted text as values.
    """
    extracted_text = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        with zip_ref.open(file_info) as file:
                            image_data = Image.open(io.BytesIO(file.read()))
                            text = pytesseract.image_to_string(image_data)
                            extracted_text[file_info.filename] = text
                    except Exception as img_error:
                        print(f"Error processing file {file_info.filename}: {img_error}")
    except Exception as e:
        print(f"Error extracting text from zip: {e}")
    return extracted_text

# Save extracted text to a CSV file
def save_text_to_csv(extracted_text, output_csv):
    """
    Saves extracted text to a CSV file.
    :param extracted_text: Dictionary with filenames as keys and extracted text as values.
    :param output_csv: Path to the output CSV file.
    """
    try:
        df = pd.DataFrame(list(extracted_text.items()), columns=['Filename', 'Extracted Text'])
        df.to_csv(output_csv, index=False)
        print(f"Extracted text saved to {output_csv}")
    except Exception as e:
        print(f"Error saving text to CSV: {e}")

# Main function to extract text from zip and save to CSV
def main():
    zip_path = '/Users/joannamenghamal/myvenv/vision_exercise/L05_DL_Vision_Receipts.zip'  # Replace with your zip file path
    output_csv = 'extracted_text.csv'  # Output CSV file path
    extracted_text = extract_text_from_zip(zip_path)
    if extracted_text:
        save_text_to_csv(extracted_text, output_csv)
    else:
        print("No text extracted from zip file.")

if __name__ == "__main__":
    main()
