# # Use a pipeline as a high-level helper
# from transformers import pipeline

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-405B-Instruct")
# pipe(messages)

# Not yet accepted: 

# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="viss-ai/LLama3-70B-SWE-LLM")
pipe(messages)