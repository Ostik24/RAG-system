from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document as LCDocument


def embed_articles(chunks, api_key):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="SEMANTIC_SIMILARITY",
        google_api_key=api_key
    )

    lc_docs = [
        LCDocument(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    vectordb = Chroma.from_documents(
        documents=lc_docs,
        embedding=embedding_model,
        persist_directory="./database"
    )

    return vectordb
