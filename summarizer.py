from transformers import BartForConditionalGeneration, BartTokenizer
from nltk import sent_tokenize
import nltk
import torch
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI
# Ensure proper download of punkt
nltk.download('punkt_tab')

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



def format_summary_openai(summary, api_key="sk-proj-KEY", model="gpt-4o-mini"):
        
        client = OpenAI(api_key=api_key)

        system_prompt= """
             # CONTEXTE
            Tu es un assistant intelligent spécialisé dans la mise en forme et le formatage de texte en MARKDOWN.

            # TACHE
            Tu vas recevoir en input le résumé d'un cours en français. Ta tâche est de convertir ce résumé en une version bien structurée au format MARKDOWN, avec des éléments tels que :

            - Une introduction claire si nécessaire.
            - Des listes à puces pour les points clés.
            - Une séparation des sections si le résumé couvre plusieurs sujets.
            - Une présentation propre et facile à lire.

            Conserve le contenu tel qu'il est, sans modifier le sens ni ajouter d'informations. Tout doit être en français.

            # EXEMPLE
            ## La Seconde Guerre mondiale

            ### Introduction
            La Seconde Guerre mondiale (1939-1945) est un conflit majeur de l'histoire moderne, impliquant la plupart des nations du monde, réparties entre les Alliés et l'Axe.

            ### Origines du conflit
            - Traité de Versailles et ses conséquences.
            - Montée en puissance de régimes totalitaires (Hitler, Mussolini, etc.).
            - Expansionnisme du Japon en Asie.

            ### Événements clés
            - 1939 : Invasion de la Pologne par l'Allemagne.
            - 1941 : Attaque de Pearl Harbor par le Japon.
            - 1944 : Débarquement en Normandie.
            - 1945 : Capitulation de l'Allemagne et du Japon.

            ### Conséquences
            - Destruction massive et pertes humaines importantes.
            - Création de l'Organisation des Nations unies (ONU).
            - Début de la Guerre froide.

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
