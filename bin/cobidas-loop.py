import json

import dotenv

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.chat_models import ChatOllama

# set up API key(s)?
dotenv.load_dotenv()

# load the cobidas questions stored by domain
qfile = '/home/bcmcpher/Projects/brainhack2023/reviewer2go/bin/cobidas-questions.json'
with open(qfile) as f:
    qcobidas = json.load(f)

# the article to load
pdf = "/home/bcmcpher/Projects/brainhack2023/articles/pdfs/Wang_2022.pdf"
loader = GenericLoader.from_filesystem(pdf, parser=PyPDFParser())

# create the reference data for the llm to search
docs = loader.load()

# split the document into chunks of reference text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# store the article in the vectorstore
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# vectorstore = Chroma.from_documents(documents=splits,
#                                     embedding=OllamaEmbeddings(model="llama3"))

# retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


# a fxn to merge document context together?
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# set up the model


# select and initialize your model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
# llm = ChatOllama(model="llama3")


# build the chain to ask the questions

# "chain" the components together to make a RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# catch the model outputs
rag_responses = []

# for each cobidas domain
for domain in qcobidas.keys():

    print(f"Asking questions about domain: {domain}")
    dqs = qcobidas[domain]

    # create an empty list for each domains extracted responses
    dcontext = []

    # for every question in the domain
    for dq in dqs:

        print(f"{domain}: {dq}")

        # invoke the chain on the question
        #dcontext.append(rag_chain.invoke(f"For the current embedded context, please answer to the best of your ability the following question: {dq}"))
