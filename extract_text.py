import pytesseract
from PIL import Image
import cv2
import os
import requests

pytesseract.pytesseract.tesseract_cmd = r"E:/tesseract_windows/tesseract.exe"

def extract_text_from_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale (better for OCR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional: apply thresholding or noise removal
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Save temp image (optional)
    cv2.imwrite("preprocessed.png", gray)

    # Run OCR
    text = pytesseract.image_to_string(gray, lang='eng')

    return text

def scan_file(filepath):
    url = "https://www.virustotal.com/api/v3/files"
    headers = {"x-apikey": os.getenv("VIRUS_TOTAL_API_KEY")}
    with open(filepath, "rb") as f:
        response = requests.post(url, headers=headers, files={"file": f})
    return response.json()