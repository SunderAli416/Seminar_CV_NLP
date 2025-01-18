
from paddleocr import PaddleOCR, draw_ocr

img_path = 'data/image/AMBER_920.jpg'
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to load model into memory

result = ocr.ocr(img_path, det=False, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)