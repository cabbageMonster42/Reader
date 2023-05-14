import os
import sqlite3
import gradio as gr
import numpy as np
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import openai
import pickle

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global variables for embeddings and cleaned texts
embeddings = None
cleaned_texts = None

# SQLite database setup
conn = sqlite3.connect('embeddings.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS embeddings
             (url TEXT PRIMARY KEY, tags TEXT, embeddings BLOB, cleaned_texts BLOB)''')

def get_content(url, tags):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    content = []
    for tag in tags:
        elements = soup.find_all(tag)
        for element in elements:
            text = ' '.join(element.stripped_strings)
            content.append(text)
    return content

def get_embedding(texts, model="text-embedding-ada-002"):
    cleaned_texts_local = [text.replace("\n", " ") for text in texts if text.strip()]
    try:
        response = openai.Embedding.create(input=cleaned_texts_local, model=model)
    except openai.error.InvalidRequestError as e:
        invalid_input = e.args[0].split(" - ")[0]
        print(f"Invalid input: {invalid_input}")
        return None
    embeddings = {i: entry['embedding'] for i, entry in enumerate(response['data'])}
    return embeddings, cleaned_texts_local

def get_embeddings_db(url, tags):
    tags_str = ','.join(tags)
    conn = sqlite3.connect('embeddings.db')  # Create a new connection here
    c = conn.cursor()
    c.execute("SELECT embeddings, cleaned_texts FROM embeddings WHERE url=? AND tags=?", (url, tags_str))
    result = c.fetchone()
    conn.close()  # Close the connection here
    if result is not None:
        embeddings, cleaned_texts = pickle.loads(result[0]), pickle.loads(result[1])
        return embeddings, cleaned_texts
    else:
        content = get_content(url, tags)
        if not content:
            return None, None
        cleaned_texts = [text.replace("\n", " ") for text in content if text.strip()]
        if not cleaned_texts:
            return None, None
        embeddings, cleaned_texts = get_embedding(cleaned_texts)
        return embeddings, cleaned_texts
    
def find_similar_embeddings(query, embeddings):
    if embeddings is None:
        print("Error: Embeddings are not available")
        return []
    query_embedding = get_embedding([query])[0][0]
    similarities = [(i, np.dot(np.array(query_embedding), np.array(doc_embedding))) for i, doc_embedding in embeddings.items()]

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [i for i, similarity in similarities[:5]]  # return the indexes of the top 5 most similar documents

def chatbot(input_text, embeddings, texts):
    similar_docs_indexes = find_similar_embeddings(input_text, embeddings)
    context = "\n".join(texts[i] for i in similar_docs_indexes)
    input_text = input_text + "\n"
    prompt = f"Your task is to read the context (which is content from a website, so you should avoid words like 'facebook ' or cookies or other useless buttons. You will only answer if the question is about the context. Here is some context:\n{context}\nUser: {input_text}, "
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

def chat_interface(url, tags, question, clear=False):
    global embeddings, cleaned_texts, c, conn

    try:
        conn = sqlite3.connect('embeddings.db')
        c = conn.cursor()
        tags_str = ','.join(tags)
        
        # Clear existing entries if required
        if clear:
            c.execute("DELETE FROM embeddings WHERE url=? AND tags=?", (url, tags_str))
            conn.commit()

        # Try to retrieve embeddings and cleaned_texts from the database
        embeddings, cleaned_texts = get_embeddings_db(url, tags)

        if embeddings is None or cleaned_texts is None:
            return "Error: Failed to generate embeddings"

        # Store the embeddings and cleaned_texts in the database
        c.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?)", (url, tags_str, pickle.dumps(embeddings), pickle.dumps(cleaned_texts)))
        conn.commit()

        # Generate a chatbot response
        answer = chatbot(question, embeddings, cleaned_texts)

    except Exception as e:
        # Handle any unexpected exceptions
        answer = f"Error: {str(e)}"
    finally:
        # Always ensure the database connection is closed, even if an error occurs
        if conn:
            conn.close()  

    return answer
# Set up the input and output interfaces
url_input = gr.inputs.Textbox(label="URL")
tags_input = gr.inputs.Textbox(label="HTML Tags (comma-separated)")
question_input = gr.inputs.Textbox(label="Question")
output_text = gr.outputs.Textbox(label="Answer")

# Create the chat interface
chat_interface = gr.Interface(
    fn=chat_interface,
    inputs=[url_input, tags_input, question_input],
    outputs=output_text,
    title="Chatbot",
    description="Ask any question based on a webpage",
    theme="compact"
)

if __name__ == '__main__':
    chat_interface.launch()
    