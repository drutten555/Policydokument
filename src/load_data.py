import os
import argparse

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import chromadb


def main(
    documents_directory: str = "documents",
    collection_name: str = "documents",
    persist_directory: str = "../db",
) -> None:
    
    # Instantiate a persistent chroma client in the persist_directory.
    # Learn more at docs.trychroma.com
    client = chromadb.PersistentClient(path=persist_directory)

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(name=collection_name)
    count = collection.count()
    print(f"Collection already contains {count} documents")

    # Read all files in the data directory
    print("=> Loading documents...")
    loader = PyPDFDirectoryLoader(documents_directory)
    documents = loader.load()

    # Extract metadata
    unique_documents = []
    for document in documents:
        results = collection.get(
            where={"source": document.metadata["source"]},
            include=["metadatas"],
        )
        if not results["ids"]:
            file_name = document.metadata["source"].split("/")[-1]
            document.metadata.update({"file_name": file_name})
            unique_documents.append(document)

    if not unique_documents:
        print("=> No new documents to be added")
        print("=> Exiting...")
        exit()

    # Instantiate the embedding model
    embedder = OllamaEmbeddings(model="nomic-embed-text")

    # Split the documents to chunks
    print("=> Splitting into chunks...")
    text_splitter = SemanticChunker(embedder)
    documents = text_splitter.split_documents(unique_documents)
    
    # Instantiate a persistent Chroma vectorstore in the persist_directory.
    db = Chroma.from_documents(
        unique_documents,
        embedder,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    new_count = collection.count()
    print(f"Added {new_count - count} documents")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Add arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="src/documents/research",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="research",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="../db",
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_dir,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
    )