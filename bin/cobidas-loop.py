import json
from pathlib import Path

import dotenv

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker

from langchain_chroma import Chroma

# from langchain import hub

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.boolean import BooleanOutputParser

# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field, field_validator

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.chat_models import ChatOllama

#
# set up
#

# set up API key(s)?
dotenv.load_dotenv()


# a fxn to merge context text blocks together
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#
# load and split a document
#

# load the cobidas questions stored by domain
qfile = '/home/bcmcpher/Projects/brainhack2023/reviewer2go/bin/cobidas-questions.json'
with open(qfile) as f:
    qcobidas = json.load(f)

# the article to load
pdf = "/home/bcmcpher/Projects/brainhack2023/articles/pdfs/Wang_2022.pdf"
loader = GenericLoader.from_filesystem(pdf, parser=PyPDFParser())

# create the reference data for the llm to search
docs = loader.load()

# split the document along these breaks
separators =  [
    "\n\n",
    "\n",
    ".",
    ",",
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    " ",
    ]

# split text based on chunk sizes
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200,
                                               separators=separators)

# try this eventually
# text_splitter = SemanticChunker(OpenAIEmbeddings())

splits = text_splitter.split_documents(docs)

#
# build a vector database from document
#

# store the article in the vectorstore
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# vectorstore = Chroma.from_documents(documents=splits,
#                                     embedding=OllamaEmbeddings(model="llama3"))

# retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

#
# build prompts
#

# generic rag prompt
# prompt = hub.pull("rlm/rag-prompt")

# rewrite the COBIDAS value as a good prompt
prompt_redo = PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Please take the the following text and restate it as a question that can be answered by an LLM with RAG context. Be precise in the details of the question so the model can easily determine if the information exists in the provided context. Restate the imporance of the information in a single sentence before restating the prompt as a question.
Prompt: {prompt}
""")

# create the boolean prompt
prompt_bool = PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Please use the following pieces of retrieved context to answer the question. You are evaluating whether their is sufficient information within the context to accurately answer the question. Reply with 'YES' if their is enough information to accurately answer the question or 'NO' if there is not enough information to accurately answer the question.
Question: {question}
Context: {context}
""")

# create the question extraction prompt
prompt_question = PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Please use the following pieces of retrieved context to answer the question. If the answer to the question is not found in the context, say that this aspect was not reported. If you don't know the answer, just say that you don't know. Answer as conciesly as possible.
Question: {question}
Context: {context}
""")

#
# set up the model
#

# select and initialize your model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
# llm = ChatOllama(model="llama3")

#
# "chain" the components together to make a RAG
#

# build from smaller elements to make context for retrieved answers available
# question_answer_chain = create_stuff_documents_chain(llm, prompt_question)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# small prompt to just restate info as a question
rag_redo = (
    prompt_redo
    | llm
    | StrOutputParser()
)

# a langchain RAG to determine if there's enough info in the first place
rag_binary = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_bool
    | llm
    | BooleanOutputParser()
)

# a langchain RAG to answer arbitrary questions
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_question
    | llm
    | StrOutputParser()
)

#
# extract the data from the document
#

# catch the model outputs
rag_responses = []

# for each cobidas domain
print(f"Extracting COBIDAS data for document: {Path(pdf).name}")
for domain in qcobidas.keys():

    print(f" -- Asking questions about domain: {domain}")
    dqs = qcobidas[domain]

    # create an empty list for each domains extracted responses
    dcontext = []

    # for every question in the domain
    for dq in dqs[:3]:  # for the first 3 question in each domain

        print(f" --  -- : {dq}")

        # reformat the COBIDAS schema as a question
        fdq = rag_redo.invoke(dq)

        # ask if there is sufficient information to answer the question
        sask = rag_binary.invoke(fdq)

        # if there is enough info, write an answer
        if sask:
            rout = rag_chain.invoke(fdq)
        # otherwise fill in None
        else:
            rout = None

        # catch the domain questions
        rag_responses.append([domain, dq, fdq, sask, rout])
