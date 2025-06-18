import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration and Setup ---
st.set_page_config(page_title="Chat with PDF using Gemini", layout="wide")

# Load environment variables from .env file
load_dotenv()

# --- Helper Functions ---

@st.cache_resource
def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Creates and stores a FAISS vector store from text chunks."""
    if not text_chunks:
        st.error("No text chunks to process. Cannot create vector store.")
        return

    try:
        # Configure the Google Generative AI with the API key
        genai.configure(api_key=api_key)
        
        # Create embeddings using Google's model
        # CORRECT
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create FAISS vector store from the text chunks and embeddings
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save the vector store in Streamlit's session state for persistence
        st.session_state.vector_store = vector_store
        st.success("PDFs processed and embeddings created successfully!")

    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        st.stop()

def get_conversational_chain(api_key):
    """Creates a conversational Q&A chain using the Gemini Pro model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, just say, "The answer is not available in the provided context."
    Do not provide a wrong or fabricated answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    # Initialize the Gemini 1.5 Flash model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    
    # Create a prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load the Q&A chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question, api_key):
    """Handles user questions and displays the model's response."""
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("Please upload and process your PDF documents first.")
        return

    try:
        vector_store = st.session_state.vector_store
        
        # Perform a similarity search to find relevant document chunks
        docs = vector_store.similarity_search(user_question)
        
        # Get the conversational chain and run it
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Display the response
        st.write("### Reply:")
        st.write(response["output_text"])

    except Exception as e:
        if "API key not valid" in str(e):
            st.error("Your Google API Key is not valid. Please check it in the sidebar.")
        else:
            st.error(f"An error occurred: {e}")

# --- Main Streamlit Application ---

def main():
    st.header("Chat with PDF using Gemini Pro 💬")

    # Attempt to get the API key from the environment file first
    api_key = os.getenv("GOOGLE_API_KEY")

    # Main content area for Q&A
    st.markdown("### Ask a Question from the PDF Files")
    user_question = st.text_input("", placeholder="Type your question here...", key="user_question")

    # Sidebar setup
    with st.sidebar:
        st.title("Menu")

        # Only show the API key input if it's not in the .env file
        if not api_key:
            st.warning("Google API Key not found in .env file.")
            api_key = st.text_input(
                "Please enter your Google API Key:", type="password", key="api_key_input"
            )
            st.markdown("""
            ---
            **Get your free API key:**
            - Visit [Google AI Studio](https://makersuite.google.com/app/apikey).
            - Ensure the 'Generative Language API' is enabled for your project in the Google Cloud Console.
            """)
        else:
            st.success("API Key loaded successfully from .env file.")

        st.title("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here and click 'Process'", accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not api_key:
                st.error("A Google API Key is required to proceed.")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks, api_key)
                    else:
                        st.error("No text could be extracted from the uploaded PDF(s).")

    # Handle user input after the sidebar is configured
    if user_question:
        if not api_key:
            st.error("Please enter your Google API Key in the sidebar to ask a question.")
        else:
            handle_user_input(user_question, api_key)

if __name__ == "__main__":
    main()