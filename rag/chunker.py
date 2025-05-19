"""
This module contains a function to chunk articles from a list of issues.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

def auto_chunk_articles(issues, chunk_size=640, chunk_overlap=100, length_threshold=500):
    """
    Automatically chunk articles from a list of issues.
    It splits the content of each article into smaller chunks based on the specified chunk size.
    If the content is smaller than the length threshold, it is added as a single chunk.
    Otherwise, it is split into smaller chunks.
    Args:
        issues (list): A list of issues, where each issue contains articles.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
        length_threshold (int): The minimum length of content to be considered for chunking.
    Returns:
        list: A list of dictionaries, each containing a chunk of content and its metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    chunk_id = 0

    for issue in issues:
        issue_title = issue.get("title")
        issue_url = issue.get("url")

        for article in issue.get("articles", []):
            article_title = article.get("title")
            content = article.get("content", "").strip()
            image_url = article.get("image", None)

            if not content:
                continue

            word_count = len(content.split())

            if word_count <= length_threshold:
                all_chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "content": content,
                    "metadata": {
                        "article_title": article_title,
                        "issue_title": issue_title,
                        "issue_url": issue_url,
                        "image_url": image_url
                    }
                })
                chunk_id += 1

            else:
                split_chunks = splitter.split_text(content)
                for chunk in split_chunks:
                    all_chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "content": chunk,
                        "metadata": {
                            "article_title": article_title,
                            "issue_title": issue_title,
                            "issue_url": issue_url,
                            "image_url": image_url
                        }
                    })
                    chunk_id += 1

    return all_chunks
