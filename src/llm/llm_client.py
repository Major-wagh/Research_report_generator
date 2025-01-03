import os
from llama_index.llms.groq import Groq

class LLMClient:
    def __init__(self):
        self.llm = Groq(model="llama3-70b-8192",api_key=os.environ.get("GROQ_API_KEY"))