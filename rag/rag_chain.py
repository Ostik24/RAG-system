"""
This module contains the function to answer a query using a RAG (Retrieval-Augmented Generation) approach.
"""
from rag.article_parser import extract_articles
from rag.chunker import auto_chunk_articles
from rag.embedder import embed_articles
import google.generativeai as genai

def vector_db(api: str) -> None:
    """
    Initialize the vector database.
    """
    json = extract_articles()

    chunks = auto_chunk_articles(json.values(), chunk_size=640, chunk_overlap=100, length_threshold=500)

    vectordb = embed_articles(chunks, api_key=api)

    return vectordb


def answer_query_with_rag(query: str, api: str, vectordb) -> str:
    """
    Retrieve context from Chroma and use Gemini to answer the query.
    Returns the generated answer as a string.
    Args:
        query (str): The query string to be answered.
    Returns:
        str: The generated answer.
    """
    system_prompt = (
        "You are an assistant that answers user questions using factual information extracted "
        "from articles in The Batch newsletter.\n\n"
        "If you see that titles are the same and urls as well, it means that the same article "
        "is repeated in the same issue. You should return only one answer combined of them. "
        "If you see different titles, you can return them separately.\n\n"
        "Return your answer in **strict JSON format** using the following schema:\n"
        "{"
        "  \"answers\": ["
        "    {"
        "      \"number\": <int>,"
        "      \"text\": <string>,"
        "      \"title\": <string>,"
        "      \"url\": <string>,"
        "      \"image_url\": <string>"
        "    }, "
        "    ..."
        "  ]"
        "}"
        "Make sure to:\n"
        "- Only use info from the provided context\n"
        "- Enumerate answers clearly using the \"number\" field\n"
        "- Always include the article \"title\" and \"urls\"\n"
        "- If the answer isn't available, return: { \"answers\": [\"text\":\"Sorry, try again\"] }"
    )
    docs = vectordb.similarity_search(query, k=7)

    context = "\n\n".join(
        f"- Title: {doc.metadata.get('article_title')}\n"
        f"  URL: {doc.metadata.get('issue_url')}\n"
        f"  Image URL: {doc.metadata.get('image_url')}\n"
        f"  Content: {doc.page_content.strip()}"
        for doc in docs
    )

    genai.configure(api_key=api)

    model = genai.GenerativeModel("models/gemini-1.5-flash")

    combined_prompt = (
        f"System: {system_prompt}\n\n"
        f"Use the following context to answer:\n\n{context}\n\nUser's question: {query}"
    )

    messages = [{"role": "user", "parts": [combined_prompt]}]

    response = model.generate_content(messages)

    return response.text
