
import os
from langchain.llms import OpenAI, Ollama

# initialize openai (remote)
llm1 = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# initialize Ollama mistral (local)
llm2 = Ollama(model="mistral")

# figure out a question
prompt = "Describe the history of AI in 3 sentences."

# pass to each model to compare
llm1(prompt)
llm2(prompt)
