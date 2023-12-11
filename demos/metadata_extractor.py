
import os
import json
from typing import Literal

from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from langchain.document_transformers.openai_functions import create_metadata_tagger

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#
# the "data" that can be extracted
#

# a list of documents (text) w/ optional metadata
original_documents = [
    Document(
        page_content="Review of The Bee Movie\nBy Roger Ebert\n\nThis is the greatest movie ever made. 4 out of 5 stars."
    ),
    Document(
        page_content="Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.",
        metadata={"reliable": False},
    ),
]

# build the json structure of data to extract
schema = {
    "properties": {
        "movie_title": {"type": "string"},
        "critic": {"type": "string"},
        "tone": {"type": "string", "enum": ["positive", "negative"]},
        "rating": {
            "type": "integer",
            "description": "The number of stars the critic rated the movie",
        },
    },
    "required": ["movie_title", "critic", "tone"],
}


# set up a pydantic class to extract typed features
class Properties(BaseModel):
    movie_title: str
    critic: str
    tone: Literal["positive", "negative"]
    rating: int = Field(description="Rating out of 5 stars")

#
# the code that extracts "metadata"
#

# must be a model that supports functions
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                 temperature=0,
                 model="gpt-3.5-turbo")

# almost possible with Ollama
llm2 = ChatOllama(model="mistral")

# load and transform the documents
document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)

# prepare the documents to get parsed
enhanced_documents = document_transformer.transform_documents(original_documents)

# print the documents to see the structure
print(*[d.page_content + "\n\n" + json.dumps(d.metadata) for d in enhanced_documents], sep="\n\n---------------\n\n")

#
# use good typing fields
#

# create the feature extractor from json object and llm
document_transformer = create_metadata_tagger(Properties, llm)

# extract the features from the documents
enhanced_documents = document_transformer.transform_documents(original_documents)

print(
    *[d.page_content + "\n\n" + json.dumps(d.metadata) for d in enhanced_documents],
    sep="\n\n---------------\n\n",
)

# use this prompt to extract the schema
prompt = ChatPromptTemplate.from_template(
"""Extract relevant information from the following text.
Anonymous critics are actually Roger Ebert.

{input}
"""
)

# create the feature extractor from json object and llm
document_transformer = create_metadata_tagger(schema, llm, prompt=prompt)

# extract the features from the documents
enhanced_documents = document_transformer.transform_documents(original_documents)

# print them
print(*[d.page_content + "\n\n" + json.dumps(d.metadata) for d in enhanced_documents], sep="\n\n---------------\n\n")
