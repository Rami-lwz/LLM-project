from flask import Flask, request, jsonify
from summarizer import resume
from summarizer import resume_formated
from summarizer import format_summary_openai
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
    summary = resume(text, max_lenght, min_length)
    return jsonify({'summary': summary})


@app.route('/summarize/pretty', methods=['POST'])
def summarize_pretty():
    data = request.get_json()
    text = data.get('text', '')
    max_lenght = data.get('max_length', 130)
    min_length = data.get('min_length', 30)
    openai_api_key= data.get('openai_api_key', None)
    summary_pretty = resume_formated(text, max_lenght, min_length, format_summary_openai, openai_api_key)
    return jsonify({'summary_pretty': summary_pretty})

if __name__ == '__main__':
    app.run(debug=False)