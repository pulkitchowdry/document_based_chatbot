import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient  # Updated import

# Load environment variables from .env file
load_dotenv()

# Initialize Hugging Face Inference Client
def load_mistral_model():
    # Create an Inference Client object with your API token
    inference = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_TOKEN"))
    return inference

# Get a response from the Mistral model
def get_mistral_response(question, inference):
    model_name = "mistralai/Mistral-Nemo-Instruct-2407"  # Specify the model name here
    # Use the inference.client method to call the model
    question = [{
        "role":"user",
        "content":question
    }
    # {
    #     "role":"user",
    #     "content":"Tell me a story"
    # }
    ]
    response = inference.chat.completions.create(
        model=model_name, 
        messages=question, 
        temperature=0.5,
        max_tokens=1024,
        top_p=0.7
    )
    return response["choices"][0]["message"]["content"]

# Initialize Streamlit app
st.set_page_config(page_title="Mistral Q&A Demo")
st.header("Chat with AI")

# Load the Mistral model
inference = load_mistral_model()

input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask a question")

# When the button is clicked
if submit and input_text:
    response = get_mistral_response(input_text, inference)
    st.subheader("Response is ")
    st.write(response)
