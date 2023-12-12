
import os
import json
import argparse

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import GrobidParser

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain import hub
from langchain.chains import RetrievalQA

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#
# build the question prompt
#

# the initial question to summarize the feature (sample size)
question = "Ignoring any behavioral study. How many subjects participated in the MRI study?"

# the template string for the structured data extraction (refinement)
template_string = """
You are a data extrator who specializes in pulling structured data from a document.
You report only what is required in the format requested.
If you do not know the answer do not guess, fill in 0.

The excerpt below delimited by triple backticks contains part of a manuscipt.
Only describe the magnetic resonance imaging (MRI) study.
Do not describe any behavioral study.
We want to know the following information about the MRI study:
1. How many participants were recruited?
2. How many participants consented to participate?
3. How many participants refused to participane?
4. How many participants were excluded from the analysis?
5. How many participants were included in the analysis?

excerpt: ```{manuscript_chunk}```

{format_instructions}
"""


# define pydantic class - camel case names, no _ in field names
class SampleSize(BaseModel):
    subjApproached: int = Field(description="The number of subjects approached")
    subjConsented: int = Field(description="The number of subjects who gave consent")
    subjRefused: int = Field(description="The number of subject who refused to participate")
    subjExcluded: int = Field(description="The number of subjects who were excluded")
    subjAnalyzed: int = Field(description="The number of subjects who were analyzed")

    @field_validator('subjAnalyzed')
    def check_score(cls, field):
        if field < 0:
            raise ValueError("Badly formed Score")
        return field


# set up parser for the defined class
pydantic_parser = PydanticOutputParser(pydantic_object=SampleSize)

#
# set up the models / embedding
#

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
llm_chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

#
# run the process
#


def main():

    # make it a cli function
    parser = argparse.ArgumentParser(description="Use a RAG to find sample size.")
    parser.add_argument("--pdf", type=str, required=True, help="The pdf to extract.")
    parser.add_argument("--out", type=str, required=True, help="The output .json file that stores the extracted data.")

    print("Extracting data from a the pdf...")

    # extract argument
    args = parser.parse_args()
    pdf = args.pdf
    out = args.out
    print(f" -- Attempting to load file: {pdf}")

    # load with the fancy loader
    loader = GenericLoader.from_filesystem(pdf, parser=GrobidParser(segment_sentences=False))

    # create the object
    pages = loader.load()

    # create object to split data into chunks for vector embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=250)

    # split the data
    all_splits = text_splitter.split_documents(pages)
    print(f" -- Split data into {len(all_splits)} chunks.")

    print(" -- Embedding splits with OpenAIEmbeddings into Chroma...")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

    print(f"Loaded {len(pages)} documents.")

    # reuse a preconfigured RAG prompt
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

    print(f"Loaded LLM model: {llm.model_name}")

    # retrieval from chain assumes at least 4 documents in vector store
    # this _should_  work
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    # the hardcoded question of returning headlines
    # question = f"Ignoring any behavioral study. How many subjects participated in the MRI study?"
    result = qa_chain({"query": question})

    print(result['result'])

    #
    # create the pydantic parsed output
    #

    # build the prompt template
    prompt = ChatPromptTemplate.from_template(template=template_string)

    # get the formatting instructions from the pydantic object
    format_instructions = pydantic_parser.get_format_instructions()

    # parse the prompt w/ the input
    messages = prompt.format_messages(manuscript_chunk=result['result'], format_instructions=format_instructions)

    # parse get the parsed output
    # output = llm.invoke(messages)
    output = llm_chat(messages)

    # pull the typed object?
    extracted_subjects = pydantic_parser.parse(output.content)

    # write the json file to disk
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(extracted_subjects.model_dump(mode="json"),
                  f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
