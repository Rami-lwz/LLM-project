from transformers import BartForConditionalGeneration, BartTokenizer
from nltk import sent_tokenize
import nltk
import torch
from concurrent.futures import ThreadPoolExecutor

# Ensure proper download of punkt
nltk.download('punkt')

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resume(texte, tokenizer, model, max_length=130, min_length=30):
    """
    Summarize a given text directly without chunking.
    """
    # Move model to GPU
    model = model.to(device)

    # Tokenize and move inputs to GPU
    inputs = tokenizer(texte, return_tensors="pt", truncation=True, padding="max_length", max_length=len(texte) // 2).to(device)

    with torch.no_grad():
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summarized_text

def get_chunks(texte, tokenizer, max_chunk_size=300):
    """
    Split text into chunks suitable for processing.
    """
    sentences = sent_tokenize(texte)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(tokenizer.tokenize(current_chunk + sentence)) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def process_chunk(chunk, tokenizer, model, chunk_size, max_length, min_length):
    """
    Process a single chunk using the model on GPU.
    """
    # Move model to GPU
    model = model.to(device)

    # Tokenize and move inputs to GPU
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=chunk_size).to(device)

    with torch.no_grad():
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def resume_chunked(chunks, tokenizer, model, chunk_size=300, max_length=130, min_length=30):
    """
    Summarize large text by splitting it into chunks, using GPU for processing.
    """
    # Move model to GPU once
    model = model.to(device)

    # Multithreading for processing chunks in parallel
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                process_chunk,
                chunks,
                [tokenizer] * len(chunks),
                [model] * len(chunks),
                [chunk_size] * len(chunks),
                [max_length] * len(chunks),
                [min_length] * len(chunks)
            )
        )
    print(f"Processed {len(results)} chunks.")
    return " ".join(results)
