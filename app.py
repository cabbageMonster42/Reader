import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
from bs4 import BeautifulSoup
import requests
import re

# Initialize the model and tokenizer

# tokenizer = GPT3Tokenizer.from_pretrained("gpt3")
# model = GPT3LMHeadModel.from_pretrained("gpt3")
model_name = "gpt2"  
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# model_name = "EleutherAI/gpt-neo-2.7B"  # Replace this with the desired GPT-Neo model name
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPTNeoForCausalLM.from_pretrained(model_name)



# HTML Scraping function
def scrap_url(url):
    # Scrap the HTML content from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ''
    for paragraph in paragraphs:
        content += paragraph.text
    content = re.sub(r'\s+', ' ', content)
    return content

# Function to generate GPT-3 embeddings
def create_embeddings(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]  # Get the last hidden state
    return last_hidden_states

# Function to generate GPT-3 response
def ask_gpt2(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.7, max_length=1000)
    gen_text = tokenizer.decode(gen_tokens[:, input_ids.shape[-1]:][0], clean_up_tokenization_spaces=True)
    return gen_text

# Handle extra UI inputs
def handle_extra_ui(inputs):
    if 'extra_ui' in inputs and inputs['extra_ui'] is not None:
        # Add the extra UI inputs to the prompt
        prompt = f' User uploaded: {inputs["extra_ui"].name}'
    else:
        prompt = ""
    return prompt

# Handle Scrap inputs
def handle_scrap(inputs):
    if 'scrap_on' in inputs and inputs['scrap_on']:
        # Scrap the URL and add it to the next question
        url = inputs['url']
        scrapped_text = scrap_url(url)
        return f' URL Content: {scrapped_text}'
    else:
        return ""

# Main inference function
def infer(user_input, scrap_on, url, extra_ui, chat_history):
    # Handle the extra UI and Scrap inputs
    extra_ui_prompt = handle_extra_ui({"extra_ui": extra_ui})
    scrap_prompt = handle_scrap({"scrap_on": scrap_on, "url": url})

    # Add the prompts to the user input
    user_input += extra_ui_prompt + scrap_prompt

    # Generate the GPT-3 response
    response = ask_gpt2(user_input)

    # Update the chat history
    chat_history = chat_history + f"\nUser----:\n {user_input}\nAI----:\n {response}"

    # Clear the user input field
    user_input = ""

    return user_input, chat_history

# In the above example, the `infer` function takes four arguments:

# - `user_input`: This is the user's chat input.
# - `scrap_on`: This is the boolean flag that determines whether or not to scrap a webpage.
# - `url`: This is the URL to scrap if `scrap_on` is true.
# - `extra_ui`: This is the file that the user uploaded.
# - `chat_history`: This is the current chat history.

# The function first handles any extra UI inputs and scrapping, if necessary, and adds the results to the user input. Then it generates a response from the GPT-3 model, updates the chat history, and clears the user input.

# Note that the `infer` function should return all of the values that you want to update on the Gradio interface. In this case, we're updating the user input (clearing it), and chat history. The order of the return values should match the order of the output fields in the Gradio interface.

# Here is how you can create the Gradio interface with the `infer` function:

# Create the Gradio interface
iface = gr.Interface(
    fn=infer, 
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder='Type something here...'), 
        gr.inputs.Checkbox(label='Scrap URL'),
        gr.inputs.Textbox(default='', placeholder='Enter URL here'),
        gr.inputs.File(label='Upload a file'),
        gr.inputs.Textbox(lines=20, default='', label='Chat History'),
    ], 
    outputs=[
        gr.outputs.Textbox(label='User Input'), 
        gr.outputs.Textbox(label='Chat History'),
    ]
)


iface.launch()
