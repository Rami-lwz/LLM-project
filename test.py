from ocr import PDFParser
from ocr import OCR
from summarizer import *
import warnings
warnings.filterwarnings("ignore")
ocr = OCR()
parser = PDFParser(ocr)

text = parser.parse_pdf('/home/noe/Workspace/LLM-project/data/uploads/19Thales.pdf')
print("\n\n")
print(len(text.split()))

type(resume_chunked(text, 300))





    # Save summarize_list in a txt file

    # Save the result of resume in a txt file

