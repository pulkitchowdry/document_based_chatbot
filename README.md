# Document-Based Q&A Chatbot

This chatbot allows users to ask questions based on pre-uploaded documents or upload their own documents. It leverages LangChain, Astra DB, and Hugging Face models to provide intelligent responses by analyzing and retrieving information from the documents stored in a vector database.

## Features

- **Pre-loaded Document Q&A**: Ask questions and get responses based on documents already uploaded in the system.
- **Custom Document Upload**: Upload your own PDF documents and query them in real-time.
- **Persistent Data Storage**: Documents are stored in Astra DB, ensuring fast and efficient retrieval for Q&A.
- **Seamless LLM Integration**: Uses Hugging Face models for natural language understanding and response generation.

## How It Works

1. **Upload a PDF Document**: Users can upload a PDF, and the document's content will be split, embedded, and stored in Astra DB.
2. **Ask a Question**: Users can ask questions, and the system will retrieve the most relevant answers from the uploaded documents.
3. **Real-Time Q&A**: After uploading a document, users can immediately query its contents and receive accurate answers.

## Tech Stack

- **Streamlit**: For building the user interface.
- **LangChain**: To handle document processing, embedding, and retrieval.
- **Astra DB**: For storing document embeddings as vectors.
- **Hugging Face**: For language model-based Q&A.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables for Astra DB and Hugging Face.
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Future Improvements

- Support for additional document types.
- Multi-language support for broader use cases.
- Display of uploaded file titles.
