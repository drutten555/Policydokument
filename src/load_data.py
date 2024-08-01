import argparse
import os
from typing import List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from chromadb.api.models.Collection import Collection

import chromadb
from pprint import pprint

def load_documents(documents_directory: str) -> List[Document]:
    """Loads PDF:s into a list. Renames the file name so that .PDF -> .pdf.

    Args:
        documents_directory (str): Path to directory with pdf documents.

    Returns:
        List[Document]: List of documents.
    """

    file_type = ".pdf"
    files = os.listdir(documents_directory)
    print(f"Nr. of files: {len(files)}")
    
    documents = []
    for filename in files:
        if filename.endswith(file_type.upper()):
            # Change '.PDF' to '.pdf'
            old_file_path = os.path.join(documents_directory, filename)
            filename = filename.replace(file_type.upper(), file_type)
            new_file_path = os.path.join(documents_directory, filename)
            os.rename(old_file_path, new_file_path)
        elif filename.endswith(file_type):
            # Read all files in the data directory
            try:
                loader = PyPDFLoader(os.path.join(documents_directory, filename))
                document = loader.load()
                documents.extend(document)
            except Exception as e:
                print(e, f"=> Skipping file: {filename}...", sep="\n")
    
    return documents

def extract_metadata(documents: List[Document], collection: Collection) -> List[Document]:
    """Extract metadata and check if document already exists in Chroma.

    Args:
        documents (List[Document]): _description_
        collection (Collection): _description_

    Returns:
        List[Document]: list of documents not in already loaded in Chroma.
    """
    new_documents = []
    for document in documents:
        results = collection.get(
            where={"source": document.metadata["source"]},
            include=["metadatas"]
        )
        if not results["ids"]:
            # Document not in Chroma.
            file_name = document.metadata["source"].split("/")[-2]
            document.metadata.update({"file_name": file_name})
            new_documents.append(document)
    return new_documents

def main(
    documents_directory: str = "documents", 
    collection_name: str = "default", 
    persist_directory: str = "db"
) -> None:

    # Instantiations
    client = chromadb.PersistentClient(path=persist_directory)  # A persistent chroma client in the persist_directory.
    embedder = OllamaEmbeddings(model="mxbai-embed-large")      # The embedding model
    text_splitter = SemanticChunker(embedder)                   #  The text splitter

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(name=collection_name)
    count = collection.count()
    print(f"=> Collection {collection.name} contains {count} documents.")
    # pprint(collection.get()["metadatas"], compact=True)

    directories = os.listdir(documents_directory)
    for dir in directories:
        print(f"\n--- {dir.upper()} ---")
        print("=> Loading documents...")
        documents_subdirectory = os.path.join(documents_directory, dir)
        documents = load_documents(documents_subdirectory)

        print("=> Extracting metadata...")
        new_documents = extract_metadata(documents, collection)
        if not new_documents:
            print("=> No new documents to be added.")
            continue

        # Split the documents to chunks
        print("=> Splitting into chunks...")
        new_documents = text_splitter.split_documents(new_documents)
        
        # Instantiate a persistent Chroma vectorstore in the persist_directory.
        print("=> Uploading documents into Chroma...")
        vectorstore = Chroma.from_documents(
            new_documents,
            embedder,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        new_count = collection.count()
        print(f"Added {new_count - count} chunks to {vectorstore._collection.name}")


if __name__ == "__main__":

    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Add arguments
    parser.add_argument(
        "--data",
        type=str,
        default="documents",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="policy",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="db",
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_dir,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
    )