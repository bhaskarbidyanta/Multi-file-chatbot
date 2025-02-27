import os
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Set API credentials
#load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")  # Ensure you have your Google API key in .env
model_name = "gemini-1.5-pro"
embedding_model = "models/embedding-001"

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text

# Streamlit UI
st.title("Conversational PDF Chatbot with Gemini")

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        text = extract_text_from_pdfs(uploaded_files)
        
        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_text(text)
        
        # Convert text chunks into embeddings and store in FAISS using Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=google_api_key)
        vectorstore = FAISS.from_texts(docs, embeddings)
        
        # Setup conversational chain with memory and Gemini chat model
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        st.success("PDFs processed! You can now ask questions.")
        
        # Chat Interface
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        user_input = st.text_input("Ask a question about the PDFs:")
        if user_input:
            response = qa_chain.run(user_input)
            st.session_state.chat_history.append((user_input, response))
            
        # Display chat history
        for question, answer in st.session_state.chat_history:
            st.write(f"**You:** {question}")
            st.write(f"**Bot:** {answer}")