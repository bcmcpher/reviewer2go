
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

# use a relatively default configuration
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=150,
                                               length_function=len,
                                               add_start_index=True)

# load a pdf
loader = PyPDFLoader("~/Projects/brainhack2023/articles/Wang_2022.pdf")
pages = loader.load()

# apply the splitter to make a loadable object
texts = text_splitter.split_documents(pages)

# pull from pubmed
from langchain.document_loaders import PubMedLoader

loader = PubMedLoader("chatgpt")
docs = loader.load()

docs[1].metadata
docs[1].page_content
