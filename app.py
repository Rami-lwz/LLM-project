from flask import Flask, request, jsonify
from summarizer import *
import ocr
from ocr import PDFParser
from ocr import OCR
from transformers import MBartForConditionalGeneration,MBart50Tokenizer
from ocr import BoringPDFParser

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Train Me!"

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    max_lenght = data.get('max_length', 130)
    min_length = data.get('min_length', 30)
    model_path= data.get('model', 'facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    chunks_context = get_chunks(text,300)
    summary = resume_chunked(chunks_context, tokenizer, model, 300, max_lenght, min_length)
    
    return jsonify({'summary': summary})



@app.route('/extract/pdf', methods=['POST'])
def extract_pdf():
    data = request.get_json()
    pdf = data.get('pdf_path', '')
    ocr = OCR()
    parser = PDFParser(ocr)
    text = parser.parse_pdf(pdf)
    return jsonify({'text': text})

@app.route('/test/pdf', methods=['POST'])
def test_summarize_pdf():
    data = request.get_json()
    pdf = data.get('pdf_path', '')
    max_lenght = data.get('max_length', 130)
    min_length = data.get('min_length', 30)
    ocr = OCR()
    parser = BoringPDFParser(ocr)
    text = parser.boring_parse_pdf(pdf)
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    open("text_ocr.txt","w").write(text)
    chunks_context = get_chunks(text,tokenizer,300)
    summary = resume_chunked(chunks_context, tokenizer, model, 300, max_lenght, min_length)
    return jsonify({'summary': summary,"original_length":len(text.split()),"summary_length":len(summary.split())})


if __name__ == '__main__':
    app.run(debug=False)