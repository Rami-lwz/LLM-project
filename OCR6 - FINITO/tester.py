import os
import pandas as pd
from ocr import PDFParser, OCR


if __name__ == '__main__':
    ocr  = OCR()
    parser = PDFParser(ocr)
    for i, dir, a in os.walk("./pdfs/"):
        if not i.endswith(".pdf"):
            continue
        print(i)
        text = parser.parse_pdf(pdf_path=f"./pdfs/{i}")
        print(text[:500])
        print(f'\n\n        how much backslashes : {len(text.split(r"\\"))}\n\n')