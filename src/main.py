import argparse
import os
import getpass
import chromadb

from typing import List
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from openai import OpenAI


def get_chatGPT_response(openai_client, query: str, context: List[str], model_name: str) -> str:
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
            "content": f"""I am going to ask you a question, which I would like you to answer.
                First answer the question with yes or no, and then give an explaination based only on the provided context.
                If there is not enough information in the context to answer the question, say "I am not sure", then try to make a guess.
                Respond with a short and consice answer.
            """
        },
        {
            "role": "user", 
            "content": f"""I'm a Chalmers employee and the question is {query}. Here is all the context you have: {context}"""}
    ]

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=prompt,
    )

    return response.choices[0].message.content  # type: ignore


def get_LLM():
    # Ask for LLM model
    llm = input(
        "Pick an LLM model:\n" +
        "(1) OpenAI \n" +
        "(2) Ollama\n\n" +
        "=> Input: "
    )
    match int(llm):
        case 1:
            # OpenAI
            model_name = "gpt-3.5-turbo"
            answer = input(f"Do you want to use GPT-4o Mini? (y/n) (default is {model_name}): ")
            if answer == "y":
                model_name = "gpt-4o-mini"
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass()
            openai_client = OpenAI() # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
        case 2:
            # Ollama
            model_name = "llama3.1"
            openai_client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            )
        case _:
            print("\nYou didn\"t pick a valid option. Terminating...")
            exit()
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

    # We use a simple input loop.
    while True:
        # Get the user's query
        query = input("=> Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print(f"\n=> Thinking using {model_name}...\n")

        # Query the collection to get the 5 most relevant results
        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )

        # Get the response from GPT
        response = get_chatGPT_response(openai_client, query, results["documents"][0], model_name)  # type: ignore
        print("Answer:", response)


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