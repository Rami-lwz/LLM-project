from transformers import pipeline
from openai import OpenAI

def resume(text,max_length,min_length):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

def resume_formated(text,max_length,min_length,formater,apikey=None):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return formater(summary[0]['summary'],apikey)


def format_summary_openai(summary, api_key="sk-proj-KEY", model="gpt-4o-mini"):
        
        client = OpenAI(api_key=api_key)

        system_prompt= """
            "Format the following text into structured study notes with headings, subheadings, and bullet points. Do NOT add more info than what there is already. Make it look like a real summary of a course.
            "Output:\n- Main Heading\n - Little sentence\n - Subheading 1\n    - Key Point 1\n    - Key Point 2\n  - Subheading 2\n    - Key Point 3\n"
    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary}
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature = 0.05
        )

        message_content = chat_completion.choices[0].message.content
        return(message_content)
