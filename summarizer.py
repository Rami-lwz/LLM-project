from transformers import pipeline

def resume(text,max_length,min_length):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

def format_summary(summary):
    #format the summary text to structured notes
    pass