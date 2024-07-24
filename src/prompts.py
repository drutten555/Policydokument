from langchain_core.prompts import PromptTemplate

"""
File containing different prompts based on use cases.
"""

policy = """Use the following pieces of context to answer the question at the end in Swedish.
    The context consists of a number of governing documents from a university. They are all in Swedish. 
    Your task is to act as an expert on the information that they contain.
    You are only allowed to answer questions based on the information from the documents.
    You will later be asked various questions that should be possible to answer with the contents of the documents.
    However, it might be that the question asked cannot be answered based on the documents’ information alone.
    
    If you lack information, the information is ambiguous, or the answer for any other reason is uncertain or unclear, state that “the answer is not clear” and explain why.
    For any answer you give, you are always forced to give supporting quotes and refer to the documents from which they originate.
    Break your answer up into nicely readable paragraphs.
    
    {context}

    Question: {question}

    Helpful Answer:
"""

research = """Use the following pieces of context to answer the question at the end in Swedish.
    The context consists of a number of notice documents of whether a research applicant was granted funding or not.
    Each application/document contains metadata in boxes inlcuding an id in the format 2023-XXXXX, and the feedback for each category/section with a grading.
    
    Your task is to act as an expert on the information they contain.
    You are only allowed to answer questions based on the information from the documents.
    You will later be asked various questions that should be possible to answer with the contents of the documents.
    However, it might be that the question asked cannot be answered based on the documents’ information alone.
    
    If you lack information, the information is ambiguous, or the answer for any other reason is uncertain or unclear, state that “the answer is not clear” and explain why.
    For any answer you give referring to any document, include the application id. Don't make up ids.
    Break your answer up into nicely readable paragraphs.

    {context}
    
    Question: {question}

    Helpful Answer:
"""

def build(prompt):
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.
    """
    return PromptTemplate.from_template(prompt)
