import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone


PINECONE_API_KEY = "ee88a295-d9f5-475f-86e3-8bb2767bdf9c"
PINECONE_ENV_REGION = "gcp-starter"
INDEX_NAME = "langchain-doc-index"
OPENAI_API_KEY = "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"

# pinecone.init(api_key=PINECONE_API_KEY,
#             environment=PINECONE_ENV_REGION)


def ingest_docs():
    loader = ReadTheDocsLoader("latest", features="html.parser")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n","\n\n"," ",""]
    )

    documents = text_splitter.split_documents(documents=raw_documents)

    print(f"Splited in to {len(documents)} splits")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents,embeddings,index_name = INDEX_NAME)
    print("**** loading to vector store ****")



if __name__ == "__main__":
    ingest_docs()
