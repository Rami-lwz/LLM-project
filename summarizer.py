from transformers import pipeline
import openai
def resume(text,max_length,min_length):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

def resume_formated(text,max_length,min_length,formater,apikey=None):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return formater(summary[0]['summary'],apikey)

def format_summary_openai(summary_text, openai_api_key=None):
    openai.api_key = openai_api_key
    
    # Prompt for formatting the summary into structured notes
    prompt = (
        "Format the following text into structured study notes with headings, subheadings, and bullet points:\n\n"
        f"{summary_text}\n\n"
        "Output:\n- Main Heading\n  - Subheading 1\n    - Key Point 1\n    - Key Point 2\n  - Subheading 2\n    - Key Point 3\n"
    )
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=len(summary_text),
        temperature=0.7
    )
    return response["choices"][0]["text"].strip()
