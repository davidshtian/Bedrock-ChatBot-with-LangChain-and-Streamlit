from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import CharacterTextSplitter

def index_directory(directory_path, glob_pattern="**/[!.]*"):
    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    # Load documents from directory
    loader = DirectoryLoader(directory_path, glob=glob_pattern, show_progress=True, use_multithreading=True)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save FAISS index locally
    vectorstore.save_local("faiss_index")

    return vectorstore

# Example usage
directory_path = "documents/"
vectorstore = index_directory(directory_path)
print(vectorstore.index.ntotal)