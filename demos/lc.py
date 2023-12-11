
import os

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import OpenAI, Ollama

# set up the callback manager - What does this do?
# cbman = CallbackManager([StreamingStdOutCallbackHandler()])

# initialize openai (remote)
llm1 = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# initialize Ollama mistral (local)
llm2 = Ollama(model="mistral")

# llm2 = Ollama(model="mistral", callback_manager=cbman)

# figure out a question
prompt = "Describe the history of AI in 3 sentences."

# compare the models
llm1(prompt)
llm2(prompt)
