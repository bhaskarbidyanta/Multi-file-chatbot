import os
import streamlit as st
from pymongo import MongoClient
from db import pdfs_collection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load API key
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

if not google_api_key:
    st.error("❌ Error: Google API Key is missing. Set 'GEMINI_API_KEY' in your .env file.")
    st.stop()

# LangChain Configurations
MODEL_NAME = "gemini-1.5-pro"
EMBEDDING_MODEL = "models/embedding-001"

st.title("📄 PDF Chatbot")

# Fetch PDFs from MongoDB
pdfs = list(pdfs_collection.find({}, {"_id": 1, "filename": 1, "content": 1}))

if not pdfs:
    st.error("❌ No PDFs found in MongoDB. Please upload a PDF first.")
    st.stop()

pdf_list = {str(doc["_id"]): doc.get("filename", "Unnamed PDF") for doc in pdfs}
selected_pdf_id = st.selectbox("📂 Choose a PDF", list(pdf_list.keys()), format_func=lambda x: pdf_list[x])

if st.button("📥 Load PDF"):
    pdf_data = next((doc for doc in pdfs if str(doc["_id"]) == selected_pdf_id), None)

    if not pdf_data or "content" not in pdf_data or not pdf_data["content"].strip():
        st.error("❌ Error: PDF content is empty or missing.")
        st.stop()

    # Split text into chunks for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(pdf_data["content"])

    # Store text chunks in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)
    vectorstore = FAISS.from_texts(docs, embeddings)

    # Setup conversational retrieval chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=google_api_key),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    st.success("✅ PDF loaded! You can now ask questions.")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.qa_chain = qa_chain  # Store chain in session

# Chatbot interface
if "qa_chain" in st.session_state:
    user_input = st.text_input("💬 Ask a question about the PDF:")

    if user_input:
        response = st.session_state.qa_chain.run(user_input)
        st.session_state.chat_history.append((user_input, response))

    # Display chat history
    for question, answer in st.session_state.chat_history:
        st.write(f"**You:** {question}")
        st.write(f"**Bot:** {answer}")

if st.button("Logout"):
    st.session_state.clear()
    st.switch_page("mainapp.py")