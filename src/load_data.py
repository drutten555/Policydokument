import argparse
import torch
import re

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma

PATH_DB = '../db'
COLLECTION_NAME = 'policy_collection'
DOCUMENTS_DIR = 'src/documents'

MODEL_NAME_KBLAB = 'KBLab/sentence-bert-swedish-cased'
MODEL_NAME_KB = 'KB/bert-base-swedish-cased'
MODEL_NAME_INTFLOAT = 'intfloat/multilingual-e5-large-instruct'

def split_documents(chunk_size, documents, tokenizer_name):
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """

    # We use a hierarchical list of separators specifically tailored for splitting documents
    MARKDOWN_SEPARATORS = [
        "\n\n\n\n",
        "\n\n\n",
        "\n\n",
        "\n",
        ".",
        ",",
        " ",
        "",
    ]
    # Remove all whitespaces between newlines e.g. \n \n \n \n --> \n\n\n\n
    for doc in documents:
        doc.page_content = re.sub('(?<=\\n) (?=\\n)', '', doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in documents:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

def get_embedding_model(model_name):
    """
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   # Check for CUDA enabled GPU
    return HuggingFaceEmbeddings(
        model_name=model_name, # Provide the pre-trained model's path
        model_kwargs={'device':device}, # Pass the model configuration options
        encode_kwargs={'normalize_embeddings': True} # Set `True` for cosine similarity
    )

def main(
    documents_directory: str = DOCUMENTS_DIR,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = PATH_DB,
) -> None:
    # Read all files in the data directory
    print('=> Loading documents...')
    loader = PyPDFDirectoryLoader(documents_directory)
    documents = loader.load()
    
    # Split the documents to chunks
    print('=> Splitting into chunks...')
    docs = split_documents(
        768,  # Choose a chunk size adapted to our model
        documents,
        tokenizer_name=MODEL_NAME_KBLAB,
    )

    # Instantiate a persistent Chroma vectorstore in the persist_directory.
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embedding_model(MODEL_NAME_KBLAB),
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    print(f"Added {len(docs)} chunks to ChromaDB")

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = vectorstore._client.get_or_create_collection(name=COLLECTION_NAME)

if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Add arguments
    parser.add_argument(
        "--data_directory",
        type=str,
        default=DOCUMENTS_DIR,
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=COLLECTION_NAME,
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default=PATH_DB,
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )