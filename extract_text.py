import pytesseract
from PIL import Image
import cv2

pytesseract.pytesseract.tesseract_cmd = r"E:/tesseract_windows/tesseract.exe"

# Load image
image = cv2.imread("./TempData/image1 (2).png")

# Convert to grayscale (better for OCR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: apply thresholding or noise removal
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Save temp image (optional)
cv2.imwrite("preprocessed.png", gray)

# Run OCR
text = pytesseract.image_to_string(gray, lang='eng')

print("Extracted Text:\n", text)