
import os

# from langchain.chat_models import ChatOpenAI, ChatOllama

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field  # , validator

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# pull a model
# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
llm = ChatOllama(model="llama2")


# define a class with parameters to extract
class Smoothing(BaseModel):
    fwhw: int = Field(description="The size of the smoothing kernel.")


# create a parser for the prompt
parser = PydanticOutputParser(pydantic_object=Smoothing)

# build template
template ="""
You are a data extractor looking for model parameters in the current text.

{partial_variables}

The text is: {text}
"""

# build the prompt template
np_prompt = PromptTemplate(template=template,
                           input_variables=["text"],
                           partial_variables={"format_instructions": parser.get_format_instructions()})

# langchain notation to combine these things into a query object
is_concept_present = np_prompt | llm | parser

# YOUR TEXT HERE
INPUT = """ """

# run a query
a = is_concept_present.invoke({"text": INPUT})
