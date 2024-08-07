import argparse
import os
import getpass
import chromadb

from typing import List
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from openai import OpenAI


def get_response(openai_client, query: str, context: List[str], model_name: str) -> str:
    """
    Queries the GPT API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """

    prompt = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions regarding Chalmers governing documents that will be provided.
                Always answer the question with yes/no/unclear, and then provide an explanation and quotes based only on the documents.
                If there is not enough information in the context to answer the question, then ask more questions. Do not assume anything about the user.
                Respond with a short and concise answer.
                Always provide a list of references containing the document's file name, 'Reference number/Diarienummer'.
            """
        },
        {
            "role": "user", 
            "content": f"""The question is {query}. Here is all the context you have: {context}"""}
    ]

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=0.7,
        stream=True
    )

    # return response.choices[0].message.content  # type: ignore, return this if stream=False
    return response


def get_LLM():
    # Ask for LLM model
    llm = input(
        "Pick an LLM model:\n" +
        "(1) OpenAI \n" +
        "(2) Ollama\n" +
        "=> Input: "
    )
    match int(llm):
        case 1: # OpenAI
            model_name = "gpt-4o-mini"
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass()
            openai_client = OpenAI() # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
        case 2: # Ollama
            model_name = "llama3.1"
            openai_client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            )
        case _:
            print("\nYou didn\"t pick a valid option. Terminating...")
            exit()
    print("=> Using", model_name)
    return model_name, openai_client


def main(
    collection_name: str = "policy", 
    persist_directory: str = "db"
) -> None:
    
    # Instantiate a LLM model
    model_name, openai_client = get_LLM()

    # Instantiate a persistent chroma client in the persist_directory.
    # This will automatically load any previously saved collections.
    client = chromadb.PersistentClient(path=persist_directory)

    # Get the collection.
    collection = client.get_collection(
        name=collection_name, 
        embedding_function=OllamaEmbeddingFunction(
            model_name="mxbai-embed-large",
            url="http://localhost:11434/api/embeddings",
        )
    )

    while True:
        # Get the user's query
        query = input("\n=> Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue

        # Query the collection to get the 5 most relevant results
        results = collection.query(
            query_texts=[query + "\nList the reference number/diarienummer for the documents."], 
            n_results=5, 
            include=["documents", "metadatas"]
        )

        # Get the response from LLM
        response = get_response(openai_client, query, results["documents"][0], model_name)  # type: ignore
        print("=> Answer:")
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    parser.add_argument(
        "--persist_dir",
        type=str,
        default="db",
        help="The directory where you want to store the Chroma collection",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="policy",
        help="The name of the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
    )