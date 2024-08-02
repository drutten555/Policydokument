from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import gradio as gr
import openai
import os
import chromadb


# Various models
MODEL_NAME_KBLAB = 'KBLab/sentence-bert-swedish-cased' # a Swedish-English bilingual model designed for mapping sentences and paragraphs into a dense vector space
MODEL_NAME_KB = 'KB/bert-base-swedish-cased' # a Swedish language model based on BERT, developed by the National Library of Sweden (KBLab)
MODEL_NAME_INTFLOAT = 'intfloat/multilingual-e5-large-instruct' # a multilingual text embedding model that supports 94 languages

PATH_DB = '/Users/kailashdejesushornig/Documents/GitHub/P2_Policydokument/db'
COLLECTION_NAME = 'policy'

FILE_PATH = './src/documents/Alkohol- och drogpolicy.pdf'
DIR_PATH = './src/documents'


## CHROMA
# Instantiate a persistent chroma client in the persist_directory.
# This will automatically load any previously saved collections.
client_db = chromadb.PersistentClient(path=PATH_DB)
ollama_ef_chroma = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="mxbai-embed-large",
)
# Get the collection.
collection = client_db.get_collection(name=COLLECTION_NAME, embedding_function=ollama_ef_chroma)

# print(client_db.list_collections()) # print collection name
# collection.get()["metadatas"] # print out content

## LANGCHAIN
ollama_ef_langchain = OllamaEmbeddings(model="mxbai-embed-large") # --> sub with ollama_ef
vectorstore = Chroma(
    client=client_db,
    embedding_function=ollama_ef_langchain
)
retriever = vectorstore.as_retriever()

# Define llm
llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4o-mini") #or "gpt-4o"
# llm = Ollama(model="llama3.1")

# Define the prompt
prompt = """
    #Background
    Use the following pieces of context to answer the question at the end.
    The context consists of a number of governing documents from a university. They are all in Swedish. 
    Your task is to act as an expert on the information that they contain. 
    You will later be asked various questions that should be possible to answer with the contents of the documents. 
    However, it might be that the question asked cannot be answered based on the documents’ information alone. 
    You are only allowed to answer questions based on the information from the documents.
    
    #ADDITION
    Answer with as much information as you can find. Keep in mind that some documents may be old and no longer valid. 
    If a document mentions that it replaces previous documents via its file number, take into account which document is the current valid one and which should prevail. 
    If you lack information, the information is ambiguous, or the answer for any other reason is uncertain or unclear, state that "SVARET ÄR INTE SÄKERT” and explain why.
    For any answer you give, you are always forced to give supporting quotes and refer to the documents from which they originate.
    Answer in Swedish.
    Break your answer up into nicely readable paragraphs.

    #RESPONSEFOMAT
    Start by repeating the question with a sentence.

    Provide answers in the format: 
    - Topic: (e.g. finance, recruitment)	
        - Document: full name of the document including the file extension  + [diarienummer]
            - Quote and the page it comes from, as well as an interpretation of the quotation.

    For each answer you give, you are always required to provide supporting quotes and refer to the documents from which they are derived.

    #DISCLAIMER 
    End any answer you give with "Observera att denna information är baserad på min sökning i de dokument som tillhandahålls och att jag kanske inte har hittat alla relevanta policyer eller riktlinjer. Om du är osäker på någon specifik aspekt rekommenderar jag att du kontaktar respektive avdelning på Chalmers eller andra relevanta myndigheter för förtydligande."
    
    {context}

    Question: {question}

    Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

llm_chain = LLMChain(
                  llm=llm, 
                  prompt=QA_CHAIN_PROMPT, 
                  callbacks=None, 
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None)
              
qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True)

def respond(question,history):
    return qa(question)["result"]


gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me question related to the governing documents at Chalmers", container=False, scale=7),
    title="Emilia Chatbot",
    examples=["Is it allowed to drink beer on campus?", 
              "Can I invite over a professor from the states and let Chalmers pay for his stay?", 
              "My name is Carl XVI Gustaf, can I park my Jaguar anywhere on campus?"],
    cache_examples=True,
    retry_btn=None,

).launch(share = True)