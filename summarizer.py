from transformers import pipeline
from openai import OpenAI
from transformers import BartForConditionalGeneration,BartTokenizer
from nltk import sent_tokenize
import nltk
import torch
from concurrent.futures import ThreadPoolExecutor

nltk.download('punkt_tab')


def get_chunks(texte,Tokenizer, max_chunk_size=300):
    sentences = sent_tokenize(texte) 
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(Tokenizer.tokenize(current_chunk + sentence)) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def process_chunk(chunk, Tokenizer, model, chunk_size, max_length, min_length):
    inputs = Tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=chunk_size)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    return Tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def resume_chunked(chunks, Tokenizer, model, chunk_size=300, max_length=130, min_length=30):
    summaries = []

    # Multithreading pour traiter plusieurs chunks en parallèle
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                process_chunk,
                chunks,
                [Tokenizer] * len(chunks),
                [model] * len(chunks),
                [chunk_size] * len(chunks),
                [max_length] * len(chunks),
                [min_length] * len(chunks)
            )
        )

    return " ".join(results)

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
