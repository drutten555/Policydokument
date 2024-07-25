import os
import PyPDF2
from tqdm import tqdm
from langchain_openai import ChatOpenAI
# from langchain.document_loaders import DocumentLoader
from langchain_community.document_loaders import TextLoader
from langchain.indexes import SimpleVectorIndex
from langchain.prompts import ChatPromptTemplate
import openai

# Load PDF files and extract text
def load_pdfs_from_directory(directory_path):
    if not os.path.isdir(directory_path):
        print("The provided path is not a directory.")
        return

    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    pdf_contents = {}

    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        pdf_path = os.path.join(directory_path, pdf_file)
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    content += page.extract_text()
                pdf_contents[pdf_file] = content
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")

    return pdf_contents

# Create LangChain documents from PDF contents
def create_documents_from_contents(pdf_contents):
    documents = []
    for filename, content in pdf_contents.items():
        documents.append({
            "text": content,
            "metadata": {"filename": filename}
        })
    return documents

# Main function
def main():
    directory_path = "/Users/kailashdejesushornig/Documents/GitHub/P2_Policydokument/src/documents/research"
    pdf_contents = load_pdfs_from_directory(directory_path)

    if not pdf_contents:
        print("No PDF contents loaded.")
        return

    # Initialize LangChain components
    documents = create_documents_from_contents(pdf_contents)
    document_loader = DocumentLoader.from_documents(documents)
    index = SimpleVectorIndex.from_documents(documents)
    
    # Create a ChatOpenAI instance
    chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    # transition to llm = ChatOllama(model="llama3")

    # Ask a question
    question = "What are the main findings of the research documents?"
    # question = "What is the average score for Vetenskaplig frågeställning aggregating all documents?"
    response = chat.ask(index, question)
    
    print("Response from the model:")
    print(response)

if __name__ == "__main__":
    main()



# Acurately scrape the scores from all files?  #1 with LLM end-to-end, then #2 by scraping 

# Convert response to CSV 