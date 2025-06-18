# -Chat-with-PDF-Files-Using-AI

# Description:
Developed a Streamlit-based interactive application that allows users to upload PDF files and ask contextual questions about the content. The system utilizes Google Generative AI for embeddings and conversational responses. The application processes PDF content, splits it into manageable chunks, and creates a searchable vector database using the FAISS library. This enables efficient similarity searches to fetch relevant content for answering user queries. The solution supports dynamic PDF uploads, question-answering with high accuracy, and ensures user queries are addressed based on the provided content only.

# Key Features:

* PDF Text Extraction: Reads and processes multiple PDF files to extract text.
* Text Chunking: Splits large text into manageable overlapping chunks for better AI performance.
* Vector Database: Implements FAISS for fast similarity searches on text embeddings.
* AI Integration: Utilizes Google Generative AI for embeddings and chat capabilities.
* Streamlit Interface: Provides a user-friendly interface for file uploads and question input.
* Contextual QA: Offers detailed answers strictly based on the content of uploaded PDFs.


# Technologies Used:
* Python 
* Streamlit
* PyPDF2 for PDF processing
* FAISS for vector database management
* Google Generative AI (Chat and Embeddings API)
* dotenv for environment variable management
* Pickle for vector store serialization


