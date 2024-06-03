# filename: bedrock_embedder.py  
import streamlit as st  
import numpy as np  
import os
import errno
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import BedrockEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
import PyPDF2

# Create a global variable for the embeddings
EMBEDDINGS = BedrockEmbeddings(model_id="cohere.embed-english-v3")  

def read_file(file):
    if file.type == "application/pdf":  
        pdf_reader = PyPDF2.PdfReader(file)  
        document = ""  
        for page in range(len(pdf_reader.pages)):  
            document += pdf_reader.pages[page].extract_text()  
    else:  
        document = file.getvalue().decode("utf-8")  
    return document

def index_file(uploaded_files=None):  
    if uploaded_files:  
        documents = [read_file(file) for file in uploaded_files]  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
        docs = text_splitter.create_documents(documents)  
        document_embeddings = EMBEDDINGS.embed_documents([doc.page_content for doc in docs])  
        combined_embeddings = np.array(document_embeddings)  
        if len(combined_embeddings.shape) == 1:  
            combined_embeddings = combined_embeddings.reshape(-1, 1)  
        return docs, combined_embeddings  
    else:  
        return None, None

def rag_search(prompt: str, index_path) -> list:  
    allow_dangerous = True  
    index_file_path = os.path.join(index_path, "index.faiss")
    if not os.path.exists(index_file_path):
        return [f"Index file {index_file_path} does not exist. Please follow these steps to create it: \n"
                "1. Upload the text or PDF files you want to index. \n"
                "2. Look for the 'Index' button at the bottom of the sidebar. \n"
                "3. Click the 'Index' button to index the files. \n"
                "The indexed files will be saved in the created folder and will be used as your local index."]
    db = FAISS.load_local(index_path, EMBEDDINGS, allow_dangerous_deserialization=allow_dangerous)  
    docs = db.similarity_search(prompt, k=5)  
    return docs

def search_index(prompt: str, index_path: str):  
    if prompt:  
        matching_docs = rag_search(prompt, index_path)  
        return matching_docs
    else:  
        return []

def save_index(vectorstore, index_path: str = "faiss_index"):
    vectorstore.save_local(index_path)

def index_file(uploaded_files=None, index_path=None):  
    if uploaded_files and index_path:  
        documents = [read_file(file) for file in uploaded_files]  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
        docs = text_splitter.create_documents(documents)  
        document_embeddings = EMBEDDINGS.embed_documents([doc.page_content for doc in docs])  
        combined_embeddings = np.array(document_embeddings)  
        
        if len(combined_embeddings.shape) == 1:  
            combined_embeddings = combined_embeddings.reshape(-1, 1)  

        # Create or load the vectorstore
        if os.path.exists(index_path):  
            if not os.listdir(index_path):
                print(f"Directory at {index_path} is empty. Creating new vectorstore.")
                vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
                # Save the new vectorstore
                save_index(vectorstore, index_path)
            else:
                print(f"Loading existing vectorstore from {index_path}")
                vectorstore = FAISS.load_local(index_path, EMBEDDINGS, allow_dangerous_deserialization=True)  
                vectorstore.add_documents(docs)
                # Save the updated vectorstore
                save_index(vectorstore, index_path)
        else:  
            # Check if the directory exists
            if not os.path.isdir(index_path):
                try:
                    # Create the index folder if it doesn't exist
                    print(f"Trying to create directory: {index_path}")
                    os.makedirs(index_path, exist_ok=True)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                    print(f"Directory {index_path} could not be created. Please check the path and your permissions. Error: {e}")

            print(f"Directory {index_path} does not exist. Creating new vectorstore.")
            vectorstore = FAISS.from_documents(docs, EMBEDDINGS)  
            # Save the new vectorstore
            save_index(vectorstore, index_path)

        return vectorstore, docs, combined_embeddings
    else:  
        print("No uploaded files or index path provided.")
        return None, None, None

def main():  
    index_path = "faiss_index"  

    # Initialize session state
    if "files_indexed" not in st.session_state:
        st.session_state["files_indexed"] = False

    # Always display file uploader
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

    if st.button("Index Files"):  
        if uploaded_files:  
            vectorstore, docs, combined_embeddings = index_file(uploaded_files, index_path)  
            if vectorstore is None or docs is None or combined_embeddings is None:  
                return  
            st.success(f"{len(uploaded_files)} files indexed. Total documents in index: {vectorstore.index.ntotal}")  

            # Set files_indexed to True after indexing
            st.session_state["files_indexed"] = True
        else:  
            st.error("Please upload files before indexing.")
  
    if "search_query" not in st.session_state:  
        st.session_state["search_query"] = ""  
  
    st.text_input("Enter your search query", key="search_query", on_change=search_index, args=(st.session_state["search_query"], index_path))
    # Create the placeholder at the point where you want the search results to appear
    placeholder = st.empty()
    # Move the display of search results and indexed documents to the bottom
    if "matching_docs" in st.session_state:  
        with placeholder.container():  
            formatted_docs = "\n\n---\n\n".join(str(doc) for doc in st.session_state['matching_docs'])
            st.markdown(f"## Search Results\n\n{formatted_docs}")
if __name__ == "__main__":  
    main()