import json

from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.chat import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# # a demo multi-lingual block of translated text
# class Translation(BaseModel):
#     es: str = Field(description="Spanish LLM translation of English (en).")
#     fr: str = Field(description="French LLM translation of English (en).")
#     kr: str = Field(description="Korean LLM translation of English (en).")
# # keep trying to get the output I want...


class Translation(BaseModel):
    text: str = Field(title="Translation",
                      description="LLM translation of English (en).")


# load up a useful (default) model
llm = Ollama(model="mistral")

# set up a translation I will (mostly) recognize
language = "Spanish"
query = "How many subjects were used in the study?"

# create an output parser
parser = JsonOutputParser(pydantic_object=Translation)

# # figure out a template that actually works...
# template = """
# Translate the following English (en) Query in triple quotes
# Query: \"\"\"{query}\"\"\"

# to Spanish, French, and Korean.

# Store the tranlsations in a JSON object based on the following instructions:
# {format_instructions}
# """

template = """
Translate the following phrase in triple quotes from English (en) to {language}.
\"\"\"{query}\"\"\"

Format the translation as a JSON object using the following instructions:
{format_instructions}
"""

# build the prompt and formatting
prompt = PromptTemplate(
    template=template,
    input_variables=["language", "query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# build the simple prompt / model / output parser chain
chain = prompt | llm | parser

# invoke the query to get a json object of the translations
tout = chain.invoke({"language": language, "query": query})

tout
# useful enough output

# TODO - w/ a loaded schema (jsonld), add the translated language w/ useful metadata.

#
# from a json file of languages, build a LangChain pydantic class
#

# create an example json file
data = {}
data['es'] = 'Spanish'
data['fr'] = 'French'
data['kr'] = 'Korean'

# dump it
lang_data = json.dumps(data)

# use lang_data to compose a class...

# a template that expexts a list of languages as inputs
template = """
Translate the following Query in triple quotes from English (en)
Query: \"\"\"{query}\"\"\"

To the following languages: {languages}

Format the resulting JSON using the following instructions: {format_instructions}
"""
# is multiple at once really more efficient?
