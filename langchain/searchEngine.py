from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)


file_path = "./data/Software-Architecture-Patterns.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

all_splits = text_splitter.split_documents(docs)

#print(len(all_splits))

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search("Give summarize view of all the architecture patterns mentioned in the pdf")

print(results[0].page_content)