# Project Policydokument

## Prerequisites
- Add an `db` and `output` directory outside the src directory.

## Ollama
To run RAG using llama3 do the following:
1. Download ollama at: https://ollama.com/
2. Run the following command in venv terminal. This should install the Llama3 model to the venv. 
    ```
    ollama run llama3
    ```
3. When done, exit the running model by writing `\bye` or `\exit`.

## Run Python script

`load_data.py` - to add documents to the Chroma DB. Do this once if you haven't initialized a ChormaDB previously or want to create a new collection with new documents.

The script can be run in the terminal with the argments `--data_dir`, `--collection_name` and `--persist_dir`.
```
python3 load.py --data_dir path/to/data --collection_name default --persist_dir path/to/db
```


`main.py` - to run the RAG bot.

The script can be run in the terminal with the argments `--collection_name` and `--persist_dir`.
```
python3 main.py --collection_name default --persist_dir path/to/db
```


## Useful commands

### **requirements.txt**
To quickly install all the required packages and dependencies for this repository run:
```
pip install -r requirements.txt
```

If you want to generate your own requirements file run the command below. This is useful if your venv has become problematic and you want to remove and create a new venv.
```
pip freeze > requirements.txt
```
