# ocr_utility.py
from paddleocr import PaddleOCR

# Initialize the OCR model once
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_image(image_path):
    """
    Extracts text from the given image using PaddleOCR.
    Returns a list of recognized text lines without coordinates.
    """
    result = ocr_model.ocr(image_path, cls=True)
    if None in result:
        return ['']
    # Each element in result looks like:
    # [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence) ]
    # We only need the text part from (text, confidence)

    extracted_texts = [line[1][0] for line in result[0]]
    return extracted_texts
