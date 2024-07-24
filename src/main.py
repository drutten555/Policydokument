import os
import argparse
import openai
import chromadb
import torch
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# Own py script
import prompts

PATH_DB = "../db"
COLLECTION_NAME = "default"

MODEL_NAME_KBLAB = "KBLab/sentence-bert-swedish-cased"
MODEL_NAME_KB = "KB/bert-base-swedish-cased"
MODEL_NAME_INTFLOAT = "intfloat/multilingual-e5-large-instruct"

def get_embedding_model(model_name):
    """
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"   # Check for CUDA enabled GPU
    return HuggingFaceEmbeddings(
        model_name=model_name, # Provide the pre-trained model"s path
        model_kwargs={"device":device}, # Pass the model configuration options
        encode_kwargs={"normalize_embeddings": True} # Set `True` for cosine similarity
    )

def get_chatGPT_response(query: str, vectorstore: Chroma, llm, model_name) -> str:
    """
    Queries the GPT API to get a response to the question.
    """

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompts.build(prompts.research)   #! change the prompt argument
        | llm
        | StrOutputParser()
    )
    
    if model_name != "llama3":
        with get_openai_callback() as cb:
            answer = rag_chain.invoke(query)
            print(cb)
    else:
        answer = rag_chain.invoke(query)

    return answer

def main(persist_directory: str = PATH_DB, collection_name: str = COLLECTION_NAME) -> None:
    # Ask for LLM model
    llm_model_name = input(
        "Pick an LLM model:\n" +
        "(1) OpenAI \n" +
        "(2) LLama3\n\n" +
        "Input: "
    )
    match llm_model_name:
        case "1":
            # Check if the OPENAI_API_KEY environment variable is set. Prompt the user to set it if not.
            if "OPENAI_API_KEY" not in os.environ:
                openai.api_key = input(
                    "Please enter your OpenAI API Key. You can get it from https://platform.openai.com/account/api-keys\n"
                )
            else:
                openai.api_key = OpenAI() # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
            
            # Ask what model to use
            model_name = "gpt-3.5-turbo"
            answer = input(f"Do you want to use GPT-4o Mini? (y/n) (default is {model_name}): ")
            if answer == "y":
                model_name = "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name)
        case "2":
            model_name = "llama3"
            llm = ChatOllama(model=model_name)
        case _:
            print("\nYou didn\"t pick a valid option. Terminating...")
            exit()
            
    # Instantiate a persistent chroma client in the persist_directory.
    # This will automatically load any previously saved collections.
    # Learn more at docs.trychroma.com
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get the Chroma vectorstore.
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(MODEL_NAME_KBLAB),
        client=client
    )

    try:
        # We use a simple input loop.
        while True:
            # Get the user"s query
            query = input("\nQuery: ")
            if len(query) == 0:
                print("Please enter a question. Ctrl+C to Quit.\n")
                continue
            print(f"\n=> Thinking using {model_name}...\n")

            # Get the response from GPT
            response = get_chatGPT_response(query, vectorstore, llm, model_name)  # type: ignore

            # Output, with sources
            print(response)
            print("\n")
            print("To exit, press Ctrl + c")
    except KeyboardInterrupt:
        print("\n=> Exiting script...")
        exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    parser.add_argument(
        "--persist_directory",
        type=str,
        default=PATH_DB,
        help="The directory where you want to store the Chroma collection",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=COLLECTION_NAME,
        help="The name of the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
    )
