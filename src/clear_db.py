import argparse
import chromadb
from chromadb.config import Settings

PATH_DB = '../db'

def main(persist_directory: str = ".") -> None:
    
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(allow_reset=True))
    client.reset()
    print("=> ChromaDB has been reset. The db contain no collections:", client.list_collections())

if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Reset ChromaDB"
    )

    # Add arguments
    parser.add_argument(
        "--persist_dir",
        type=str,
        default=PATH_DB,
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()
    main(
        persist_directory=args.persist_dir,
    )