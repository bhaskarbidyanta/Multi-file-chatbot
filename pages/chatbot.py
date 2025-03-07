import os
import streamlit as st
import speech_recognition as sr
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

st.title("📄 Multi-PDF Chatbot with Voice Commands")

# Fetch PDFs from MongoDB
pdfs = list(pdfs_collection.find({}, {"_id": 1, "filename": 1, "content": 1}))

if not pdfs:
    st.error("❌ No PDFs found in MongoDB. Please upload a PDF first.")
    st.stop()

pdf_list = {str(doc["_id"]): doc.get("filename", "Unnamed PDF") for doc in pdfs}
selected_pdf_ids = st.multiselect("📂 Choose PDFs", list(pdf_list.keys()), format_func=lambda x: pdf_list[x])

if st.button("📥 Load PDFs"):
    selected_pdfs = [doc for doc in pdfs if str(doc["_id"]) in selected_pdf_ids]
    
    if not selected_pdfs:
        st.error("❌ No valid PDFs selected.")
        st.stop()
    
    all_texts = []
    for pdf in selected_pdfs:
        if "content" in pdf and pdf["content"].strip():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(pdf["content"])
            all_texts.extend(chunks)
    
    if not all_texts:
        st.error("❌ Error: Selected PDFs have empty or missing content.")
        st.stop()
    
    # Store text chunks in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)
    vectorstore = FAISS.from_texts(all_texts, embeddings)
    
    # Setup conversational retrieval chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=google_api_key),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    st.success("✅ PDFs loaded! You can now ask questions.")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.qa_chain = qa_chain  # Store chain in session
#Voice recognition function
def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        try:
            audio = recognizer.listen(source,timeout=5)
            command = recognizer.recognize_google(audio)
            return command
        except sr.UnknownValueError:
            st.warning("Could not understand the voice input.")
        except sr.RequestError:
            st.warning("Speech recognition service is unavailable.")

# Chatbot interface
if "qa_chain" in st.session_state:
    col1,col2 = st.columns([3,1])

    with col1:
        user_input = st.text_input("💬 Ask a question about the PDFs:")

    with col2:
        if st.button("🎤 Use Voice (Press Spacebar)"):
            voice_query = get_voice_command()
            if voice_query:
                user_input = voice_query
                st.text(f"You said: {voice_query}")
    
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
