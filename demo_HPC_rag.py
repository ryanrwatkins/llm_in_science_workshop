#In Cerberus HPC > VS Code > Terminal
#ml python3/3.12.9
#ml ollama/0.6.8
#ollama serve &
#python3 -m venv ollama_venv  #(could create new one, or use previous)
#source ollama_venv/bin/activate
#pip install requests numpy ollama



import numpy as np
import ollama
import os
import re
import time
import json
from numpy.linalg import norm

import re

def parse_file(filename, words_per_chunk=100):
    with open(filename, encoding="utf-8-sig") as f:
        text = f.read()

    # Remove Gutenberg header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1 and end != -1:
        text = text[start + len(start_marker):end]

    # Clean up and split
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    return chunks


def save_embeddings(filename, embeddings):
    # create dir if it doesnt exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)
    print("save embeddings")

def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open (f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)['embedding']
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    print("saved embeddings")
    return embeddings

def find_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    print("Found similar")
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
                        based on the snippets of text provided in context. Answer
                        only using the context provided, being as concise as possible and quoting from the context when useful.
                        If you are unsure, just say you don't know.
                        Context:
                        """

    filename = "book.txt"
    paragraphs = parse_file("book.txt", words_per_chunk=100)

    
    embeddings = get_embeddings(filename, 'mxbai-embed-large', paragraphs)

    prompt = input("What do you want?")
    prompt_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=prompt)[
        "embedding"
        ]
    # most similar results
    most_similar_chunks = find_similar(prompt_embedding, embeddings)[:5]
    print("Top similar chunks:")
    for score, idx in most_similar_chunks:
        print(f"{score:.4f} - {idx} - {paragraphs[idx][:100]!r}")

    context = SYSTEM_PROMPT + "\n" + "\n---\n".join(paragraphs[item[1]] for item in most_similar_chunks)

    
    response = ollama.chat(
        model='llama3.1:8b',

        messages=[
            {
                "role": "system",
                #"content": SYSTEM_PROMPT +"\n".join(paragraphs[item[1]] for item in most_similar_chunks)
                "content": context
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )
    
    print(response["message"]["content"])
        

if __name__ == "__main__":
    main()
