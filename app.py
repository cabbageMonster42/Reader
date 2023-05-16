import os
import gradio as gr
import numpy as np
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import openai
import pickle
from tiktoken import tokenizer


# Load environment variables
load_dotenv()
# Read the content of your Python script
with open('test.py', 'r') as f:
    script_content = f.read()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global variables for embeddings and cleaned texts
embeddings = None
cleaned_texts = None

# Global dictionary to store saved embeddings
saved_embeddings = {}

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



def find_similar_embeddings(query, embeddings):
    query_embedding = get_embedding([query])[0][0]
    similarities = [(i, np.dot(np.array(query_embedding), np.array(doc_embedding))) for i, doc_embedding in embeddings[0].items()]

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [i for i, similarity in similarities[:5]]  # return the indexes of the top 5 most similar documents



def chatbot(input_text, embeddings, texts):

    similar_docs_indexes = find_similar_embeddings(input_text, embeddings)
    with open('readme.md', 'r') as file:
        content = file.read().strip()

    content_tokens = tokenizer.count_tokens(content)
    if content_tokens > 1000:
        # If the content has more than 1000 tokens, truncate it
        content = tokenizer.truncate_tokens(content, 1000)

    context = "\n".join(texts[i] for i in similar_docs_indexes)
    input_text = input_text + "\n"
    prompt = f"""Your task is to read the context. You are a skilled software developer. This is the app you are working on: {content} You will compose an answer in the following structure: \n 'Problem:<<10-20 words.>> \n Process:<<30-50 words or a list of bulletpoints.>> \n Implementation:<<This part must be elaborate and well a definded step by step guide: Describe how to implement, make a list of dependencies, decide architecture, create a filesystem structure etc.>> \n Final thougths:<< >>' Here is some context:\n{context}\nUser: {input_text}, """

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=1500,
            temperature=0.7,
            n=1,
            stop=None
        )
    except Exception as e:
        print(f"Failed to call OpenAI API: {e}")
        return None

    if response and response.choices and response.choices[0].text.strip():
        return response.choices[0].text.strip()

    print("No valid response from the OpenAI API.")
    return None

previous_url = None

def chat_interface(url, tags, question, save=False, clear=False, reset=False):
    global embeddings, cleaned_texts, previous_url, saved_embeddings

    # Reset the embeddings if the checkbox is selected
    if reset:
        embeddings = None
        cleaned_texts = None

    # Clear embeddings and cleaned texts if requested or if a new URL is entered
    if clear or reset or (embeddings is not None and cleaned_texts is not None and url != previous_url):
        embeddings = None
        cleaned_texts = None
    
    if reset:
        embeddings = None
        cleaned_texts = None

    # Check if the URL's embeddings are saved and load them
    if url in saved_embeddings and not clear:
        embeddings, cleaned_texts = saved_embeddings[url]
    else:
        # Get and clean content if embeddings are not available or the URL has changed
        if embeddings is None or cleaned_texts is None or url != previous_url:
            content = get_content(url, tags)
            if not content:
                return "Error: No content found"

            cleaned_texts = [text.replace("\n", " ") for text in content if text.strip()]
            if not cleaned_texts:
                return "Error: No valid text content found"

            embedding_result = get_embedding(cleaned_texts)
            if embedding_result is None:
                return "Error: Failed to generate embeddings"
            embeddings = embedding_result
            
        print("Generating embedding: ... \n...\n...")
        print("Cleaned Texts:", cleaned_texts)
        print("Cleaned Tags:\n +++\n +++\n", tags)

    # Save the embeddings if the save checkbox is ticked
    if save:
        saved_embeddings[url] = (embeddings, cleaned_texts)
        # Optionally, save the dictionary to a file so it persists across sessions
        with open('saved_embeddings.pkl', 'wb') as f:
            pickle.dump(saved_embeddings, f)

    # Chat with the assistant
    answer = chatbot(question, embeddings, cleaned_texts)

    # Store the previous URL for comparison
    previous_url = url

    return answer

# Set up the input and output interfaces
url_input = gr.inputs.Textbox(label="URL")
tags_input = gr.inputs.Textbox(label="HTML Tags (comma-separated)")
question_input = gr.inputs.Textbox(label="Question")
save_input = gr.inputs.Checkbox(label="Save embeddings")
output_text = gr.outputs.Textbox(label="Answer")
reset_input = gr.inputs.Checkbox(label="Reset Embeddings")


# Create the chat interface
chat_interface = gr.Interface(
    fn=chat_interface,
    inputs=[url_input, tags_input, question_input, save_input, reset_input],
    outputs=output_text,
    title="Chatbot",
    description="Ask any question based on a webpage",
    theme="compact"
)

if __name__ == '__main__':
   chat_interface.launch()
    