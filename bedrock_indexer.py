from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def index_directory(directory_path, glob_pattern="**/[!.]*", chunk_size=500):
    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    # Load documents from directory
    loader = DirectoryLoader(
        directory_path, 
        glob=glob_pattern, 
        show_progress=True, 
        use_multithreading=True
    )
    documents = loader.load()

    # Use RecursiveCharacterTextSplitter for better chunk handling
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    # Create and save FAISS vectorstore
    return FAISS.from_documents(docs, embeddings).save_local("faiss_index")

# Example usage
directory_path = "documents/"
vectorstore = index_directory(directory_path)
print(vectorstore.index.ntotal)