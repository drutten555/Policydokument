# Project Policydokument

## Ollama
To run RAG using llama3, you have to first download the model to your computer.
1. Download ollama at: https://ollama.com/
2. Run on of the following commands in venv terminal. This should install the Llama3 model to the venv. 
    ```
    ollama run llama3.1  # pulls and runs the LLM 
    ```
    
    or

    ```
    ollama pull llama3.1  # only pulls the LLM
    ```
3. When done then you are ready to run the scripts! If you used the `ollama run` command then exit the running model by writing `\bye` or `\exit`.

The Llama3.1 model is by default 8b and performs quite badly on answering questions using RAG. More work/prompt engineering is needed.


## Run Python script

- `load_data.py` - to add documents to the Chroma DB. Do this once if you haven't initialized a ChormaDB previously or want to create a new collection with new documents.

    The script can be run in the terminal with the argments `--data_dir`, `--collection_name` and `--persist_dir`.
    ```
    python3 load.py --data_dir path/to/data --collection_name policy --persist_dir path/to/db
    ```

- `main.py` - to run the RAG bot.
    
    The script can be run in the terminal with the argments `--collection_name` and `--persist_dir`.
    ```
    python3 main.py --collection_name policy --persist_dir path/to/db
    ```

- To reset the ChromaDB, remove everything in persistent database folder (`db`).
    ```
    $ rm -r db/*
    ```