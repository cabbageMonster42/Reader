import os
import gradio as gr
import numpy as np
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global variables for embeddings and cleaned texts
embeddings = None
cleaned_texts = None

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
    context = "\n".join(texts[i] for i in similar_docs_indexes)
    input_text = input_text + "\n"
    prompt = f"You are a helpful assistant. You will ignore any mention of cookies. You will only answer if the question is about the context.This the code now and you will modify it as I say to integrate feature in the context; Here is some context:\n{context}\nUser: {input_text}"
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

previous_url = None

def chat_interface(url, tags, question, clear=False):
    global embeddings, cleaned_texts, previous_url

    # Clear embeddings and cleaned texts if requested or if a new URL is entered
    if clear or (embeddings is not None and cleaned_texts is not None and url != previous_url):
        embeddings = None
        cleaned_texts = None

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
        
        print("After embedding generation:")
        print("Embeddings:", embeddings)
        print("Cleaned Texts:", cleaned_texts)
        

    # Chat with the assistant
    answer = chatbot(question, embeddings, cleaned_texts)

    # Store the previous URL for comparison
    previous_url = url

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

# working but EXPENSIVE >>>>>>>>>>>>>>>


# import openai
# import gradio as gr
# import numpy as np
# from bs4 import BeautifulSoup
# import requests
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')

# def get_content(url, tags):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     content = []
#     for tag in tags:
#         elements = soup.find_all(tag)
#         for element in elements:
#             text = ' '.join(element.stripped_strings)
#             content.append(text)
#     return content

# def get_embedding(texts, model="text-embedding-ada-002"):
#     cleaned_texts = [text.replace("\n", " ") for text in texts if text.strip()]
#     try:
#         response = openai.Embedding.create(input=cleaned_texts, model=model)
#     except openai.error.InvalidRequestError as e:
#         invalid_input = e.args[0].split(" - ")[0]
#         print(f"Invalid input: {invalid_input}")
#         return None
#     embeddings = {i: entry['embedding'] for i, entry in enumerate(response['data'])}
#     return embeddings, cleaned_texts

# def find_similar_embeddings(query, embeddings):
#     query_embedding = get_embedding([query])[0][0]
#     similarities = [(i, np.dot(np.array(query_embedding), np.array(doc_embedding))) for i, doc_embedding in embeddings.items()]
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return [i for i, similarity in similarities[:5]]  # return the indexes of the top 5 most similar documents

# def chatbot(input_text, embeddings, texts):
#     similar_docs_indexes = find_similar_embeddings(input_text, embeddings)
#     context = "\n".join(texts[i] for i in similar_docs_indexes)
#     input_text = input_text + "\n"
#     prompt = f"You are a helpful assistant. Here is some context:\n{context}\nUser: {input_text}"
#     response = openai.Completion.create(
#         model="text-davinci-002",
#         prompt=prompt,
#         max_tokens=50,
#         temperature=0.7,
#         n=1,
#         stop=None
#     )
#     return response.choices[0].text.strip()

# def chat_interface(url, tags, question):
#     # Get and clean content
#     content = get_content(url, tags)

#     # Get the embeddings and the cleaned texts
#     embeddings, cleaned_texts = get_embedding(content)

#     # Chat with the assistant
#     answer = chatbot(question, embeddings, cleaned_texts)

#     return answer

# # Set up the input and output interfaces
# url_input = gr.inputs.Textbox(label="URL")
# tags_input = gr.inputs.Textbox(label="HTML Tags (comma-separated)")
# question_input = gr.inputs.Textbox(label="Question")
# output_text = gr.outputs.Textbox(label="Answer")

# # Create the chat interface
# chat_interface = gr.Interface(
#     fn=chat_interface,
#     inputs=[url_input, tags_input, question_input],
#     outputs=output_text,
#     title="Chatbot",
#     description="Ask any question based on a webpage",
#     theme="compact"
# )

# if __name__ == '__main__':
#     chat_interface.launch()


# ///// working >>>>>

# import openai
# import os
# from dotenv import load_dotenv
# import requests
# from bs4 import BeautifulSoup
# import gradio as gr
# import numpy as np


# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')

# def get_content(url, tags):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     content = []
#     for tag in tags:
#         elements = soup.find_all(tag)
#         for element in elements:
#             text = ' '.join(element.stripped_strings)
#             content.append(text)
#     return content

# def get_embedding(texts, model="text-embedding-ada-002"):
#     cleaned_texts = [text.replace("\n", " ") for text in texts if text.strip()]
#     try:
#         response = openai.Embedding.create(input=cleaned_texts, model=model)
#     except openai.error.InvalidRequestError as e:
#         invalid_input = e.args[0].split(" - ")[0]
#         print(f"Invalid input: {invalid_input}")
#         return None
#     embeddings = {i: entry['embedding'] for i, entry in enumerate(response['data'])}
#     return embeddings, cleaned_texts

# def find_similar_embeddings(query, embeddings):
#     query_embedding = get_embedding([query])[0][0]
#     similarities = [(i, np.dot(np.array(query_embedding), np.array(doc_embedding))) for i, doc_embedding in embeddings.items()]
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return [i for i, similarity in similarities[:5]]  # return the indexes of the top 5 most similar documents

# def chatbot(input_text, embeddings, texts):
#     similar_docs_indexes = find_similar_embeddings(input_text, embeddings)
#     context = "\n".join(texts[i] for i in similar_docs_indexes)
#     input_text = input_text + "\n"
#     prompt = f"You are a helpful assistant. This is scraped information from my website. You will respond to any question related to:\n{context}\nUser: {input_text}"
#     response = openai.Completion.create(
#         model="text-davinci-002",
#         prompt=prompt,
#         max_tokens=200,
#         temperature=0.7,
#         n=1,
#         stop=None
#     )
#     return response.choices[0].text.strip()


# def main():
#     url = input('Please enter the URL: ')
#     tags = input('Please enter the HTML tags to scrape, separated by commas: ').split(',')

#     # Get and clean content
#     content = get_content(url, tags)

#     # Get the embeddings and the cleaned texts
#     embeddings, cleaned_texts = get_embedding(content)

#     # Prompt for questions
#     def ask_question(question):
#         return chatbot(question, embeddings, cleaned_texts)

#     # Set up the chat interface
#     chat_interface = gr.Interface(
#         fn=ask_question,
#         inputs="text",
#         outputs="text",
#         title="Chatbot",
#         description="Ask any question"
#     )

#     # Launch the chat interface
#     chat_interface.launch()


# if __name__ == '__main__':
#     main()
