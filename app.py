from flask import Flask, request, jsonify
from summarizer import resume

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

if __name__ == '__main__':
    app.run(debug=False)