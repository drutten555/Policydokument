from openai.types.chat import ChatCompletionMessageParam
from typing import List

"""
File containing different prompts based on use cases.

Can you extract the information and metadata to json?
What metadata does all documents have?
Based on the metadata you identified, can you give me an example from a document with all the metadata information?
"""

default = """I am going to ask you a question, which I would like you to answer"
        based only on the provided context, and not any other information.
        If there is not enough information in the context to answer the question,
        'say "I am not sure", then try to make a guess.'
        Break your answer up into nicely readable paragraphs.
"""

def build_prompt(query: str, context: List[str], content) -> List[ChatCompletionMessageParam]:
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    More information: https://platform.openai.com/docs/guides/chat/introduction

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (List[ChatCompletionMessageParam]).
    """

    system: ChatCompletionMessageParam = {
        "role": "system",
        "content": content,
    }
    user: ChatCompletionMessageParam = {
        "role": "user",
        "content": f"The question is {query}. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    return [system, user]
