# Environment variables

import os
from dotenv import load_dotenv

load_dotenv()

# UI

import streamlit as st

# Langchain components

from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from huggingface_hub import InferenceClient
from langchain_community.llms import huggingface_endpoint
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

# Dataset retrieval from HuggingFace

from datasets import load_dataset

# CassIO - Engine powering AstraDB integration with Langchain

import cassio

# PDF Reader

from PyPDF2 import PdfReader

# Path of the PDF document which needs to be read

# pdfreader = PdfReader("doc.pdf")

# Uploading files and reading it

st.title("Langchain + AstraDB + HuggingFace Chatbot")
st.subheader("Upload a file to train the Chatbot")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Connecting to CassIO

ASTRA_DB_API_TOKEN = os.getenv("ASTRA_DB_API_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
cassio.init(token=ASTRA_DB_API_TOKEN, database_id=ASTRA_DB_ID)

# Creating Langchain embedding and LLM objects

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    max_length=128,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if uploaded_file is not None:
    pdfreader = PdfReader(uploaded_file)


    # Extract text from the document

    from typing_extensions import Concatenate
    raw_text = ""
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Creating the Langchain vector store in Astra DB

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="demo",
        session=None,
        keyspace=None
    )

    # Divide the document into chunks

    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 800,
        chunk_overlap = 200,
        length_function = len
    )

    text = text_splitter.split_text(raw_text)

    # Upload data to Astra DB

    astra_vector_store.add_texts(text)
    print("Inserted %i lines." %len(text))
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

else:
    st.write("Upload a PDF file")

# Streamlit App UI - Q&A part

astra_vector_store_qa = Cassandra(
    embedding=embedding,  # Not needed here, just for querying
    table_name="demo",  # Use the same table as above
    session=None,
    keyspace=None
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store_qa)

st.subheader("Ask questions from the files uploaded")

query = st.text_input("Your question: ")

if st.button("Ask a question"):
    if query:
        answer = astra_vector_index.query(query, llm=llm).strip()
        st.write(f"**Answer:** {answer}")
            
    else:
        st.write("Please enter a question. ")



        