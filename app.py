import openai
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file



## Function to load OpenAI model and get a response
def get_openai_response(question):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Call the OpenAI ChatCompletion API for GPT-3.5-turbo
    response = openai.chat.completions.create(  # NOTE: This is the new format in openai>=1.0.0
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

## Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

input = st.text_input("Input: ", key="input")
submit = st.button("Ask a question")

# When the button is clicked
if submit and input:
    response = get_openai_response(input)
    st.subheader("Response is ")
    st.write(response)
