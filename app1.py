from pydantic import SecretStr
import streamlit as st
import os
import base64
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import google.generativeai as genai

# Decode the Base64 API key
def decode_api_key(encoded_api_key):
    decoded_bytes = base64.b64decode(encoded_api_key.encode('utf-8'))
    decoded_str = str(decoded_bytes, 'utf-8')
    return decoded_str

# Replace with your Base64 encoded key
api = "QUl6YVN5QXMyRUd1cjR4d3BrZW90cUtDakJPSmZLQkF2SVlZM3VN"
decoded_api_key = decode_api_key(api)
# genai.configure(api_key=decoded_api_key)  # Removed: not needed and not exported

# Title
st.title("📄 **PDF RAG Query System**")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("**Upload a PDF file**", type=["pdf"])

# Query input
query = st.text_input("**Enter your query to find relevant content from the PDF**")

# Setup directories
save_directory = "pdfs_uploaded"
embedding_dir = "embeddings_pdf"

os.makedirs(save_directory, exist_ok=True)
os.makedirs(embedding_dir, exist_ok=True)

progress_text = "Processing... Please wait."

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Split into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return splitter.split_text(text)

# File Upload & Processing
if uploaded_file:
    file_path = os.path.join(save_directory, uploaded_file.name)
    if not os.path.exists(file_path):
        with st.sidebar:
            progress_bar = st.progress(0, text=progress_text)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress_bar.progress(25)

        # Extract and chunk text
        text = extract_text_from_pdf(file_path)
        chunk_text = get_text_chunks(text)
        progress_bar.progress(50)

        documents = [Document(page_content=chunk, metadata={"path": file_path}) for chunk in chunk_text]
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(decoded_api_key))
        db = FAISS.from_documents(documents, embeddings)
        progress_bar.progress(75)

        db.save_local(embedding_dir)
        progress_bar.progress(100)
        st.sidebar.success("PDF uploaded and processed successfully.")
    else:
        st.sidebar.info("PDF already exists. Ready for querying.")

# List uploaded PDFs
st.sidebar.title("Uploaded PDF Files")
for file in os.listdir(save_directory):
    with st.sidebar.expander(file):
        if st.button(f"Delete {file}", key=f"delete_{file}"):
            os.remove(os.path.join(save_directory, file))
            st.sidebar.success(f"{file} deleted.")
            st.rerun()

# Search Button
# Add this after import google.generativeai as genai
genai.configure(api_key=decoded_api_key)
llm_model = genai.GenerativeModel('gemini-1.5-flash-002')


if st.button("🔍 Search"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(decoded_api_key))
    embedding_storage = FAISS.load_local(embedding_dir, embeddings, allow_dangerous_deserialization=True)
    
    if embedding_storage and query:
        with st.spinner("Searching..."):
            docs = embedding_storage.similarity_search(query, k=4)  # top 4 chunks
            if docs:
                # Combine all retrieved chunks
                retrieved_text = "\n\n".join([doc.page_content for doc in docs])

                # Construct the final prompt
                prompt = f"""
You are a helpful assistant. Use the provided document context to answer the user's question precisely and concisely.

Context:
{retrieved_text}

Question: {query}

Answer:"""

                # Get LLM response
                try:
                    response = llm_model.generate_content(prompt)
                    answer = response.text.strip()
                except Exception as e:
                    answer = f"❌ Error from LLM: {e}"

                # Display Results
                st.subheader("🤖 LLM Response")
                st.write(answer)

                with st.expander("📄 Retrieved Chunks"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")
            else:
                st.warning("No matching content found.")
    else:
        st.error("Please upload a PDF file and enter a query.")
