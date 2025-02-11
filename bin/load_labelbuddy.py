
import json
import os
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# look at text splitters
separators = ["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
rcc = RecursiveCharacterTextSplitter(chunk_overlap=50, separators=separators, keep_separator=False)

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
mdh = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)

# path to labelbuddy data
datadir = "/home/bcmcpher/Projects/labelbuddy/labelbuddy-annotations/projects/cobidas"
documents = Path(datadir, "documents", "2023", "documents_00001.jsonl")

# load the annotations?

# load the json archive of articles
articles = []  # create empty list for parsed article text
with open(documents, "r") as json_file:  # open the .jsonl file
    for line in json_file:  # for every line (article) in the file
        jsonarticle = json.loads(line)  # parse the articles json
        textarticle = jsonarticle.get("text")  # extract the text component
        md_split = mdh.split_text(textarticle)  # split based on markdown
        article = []  # create an empty list for text chunks
        for chunk in md_split:  # for every MD section
            # check metadata - only append body txt
            article.append(rcc.split_text(chunk.page_content))  # split only the page content
        articles.append(article)  # append the page content

# test stuff

zz = articles[0][7]



# initialize the llms
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# embed an articles chunks...

llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0)
llm_chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0)

# ask questions...
