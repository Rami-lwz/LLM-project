from transformers import pipeline

def resume(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    max_length = max(1, int(len(text.split()) * 0.3))
    min_length = max(1, int(len(text.split()) * 0.2))

    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
