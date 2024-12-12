from flask import Flask, request, jsonify
from summarizer import *
import ocr
from ocr import PDFParser
from ocr import OCR
from transformers import MBartForConditionalGeneration,MBart50Tokenizer
from ocr import BoringPDFParser
from qcmGenerator import get_questions
import os
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
    model_path=data.get('model_path',"/workspace/LLM-project/models/custom_mbart_french")
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    chunks_context = get_chunks(text,tokenizer,300)
    summary = resume_chunked(chunks_context, tokenizer, model, 300, max_lenght, min_length)
    return jsonify({'summary': summary,"original_length":len(text.split()),"summary_length":len(summary.split())})

@app.route('/summarize/pretty', methods=['POST'])
def summarize_pretty():
    data = request.get_json()
    text = data.get('text', '')
    max_lenght = data.get('max_length', 130)
    min_length = data.get('min_length', 30)
    model_path=data.get('model_path',"/workspace/LLM-project/models/custom_mbart_french")
    api_key=data.get('api_key',"")
    model_openai_name=data.get('model_openai',"gpt-4o-mini")
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    chunks_context = get_chunks(text,tokenizer,300)
    summary = resume_chunked(chunks_context, tokenizer, model, 300, max_lenght, min_length)
    pretty_summary=format_summary_openai(summary,api_key,model_openai_name)
    return jsonify({'summary': pretty_summary,"original_length":len(text.split()),"summary_length":len(summary.split())})


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
    model_path=data.get('model_path',"/workspace/LLM-project/models/custom_mbart_french")
    ocr = OCR()
    parser = PDFParser(ocr)
    text = parser.parse_pdf(pdf)
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    open("text_ocr.txt","w").write(text)
    chunks_context = get_chunks(text,tokenizer,300)
    summary = resume_chunked(chunks_context, tokenizer, model, 300, max_lenght, min_length)
    return jsonify({'summary': summary,"original_length":len(text.split()),"summary_length":len(summary.split())})

@app.route('/test/pdf/pretty', methods=['POST'])
def test_summarize_pdf_pretty():
    data = request.get_json()
    pdf = data.get('pdf_path', '')
    max_lenght = data.get('max_length', 130)
    min_length = data.get('min_length', 30)
    model_path=data.get('model_path',"/workspace/LLM-project/models/custom_mbart_french")
    model_openai_name=data.get('model_openai',"gpt-4o-mini")
    api_key=data.get('api_key',"")
    ocr = OCR()
    parser = PDFParser(ocr)
    text = parser.parse_pdf(pdf)
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    open("text_ocr.txt","w").write(text)
    chunks_context = get_chunks(text,tokenizer,300)
    summary = resume_chunked(chunks_context, tokenizer, model, 300, max_lenght, min_length)
    pretty_summary=format_summary_openai(summary,api_key,model_openai_name)
    return jsonify({'summary': pretty_summary,"original_length":len(text.split()),"summary_length":len(summary.split())})


@app.route('/generate/qcm', methods=['POST'])
def generate_qcm():
    data = request.get_json()
    course = data.get('raw_text', '')
    api_key = data.get('api_key', 'sk-proj-KEY')
    model = data.get('model', 'gpt-4o-mini')
    questions = get_questions(course, api_key, model)
    return jsonify({'questions': questions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
