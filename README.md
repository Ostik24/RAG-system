# The Batch QA Assistant – RAG System

A Retrieval-Augmented Generation (RAG) system that answers user questions based on articles from **The Batch** newsletter by DeepLearning.AI. It combines document retrieval, Gemini-powered reasoning, and a simple Streamlit UI to offer structured, source-cited answers.

---

## Objective

Design and implement a multimodal RAG system that:
- Retrieves relevant newsletter articles
- Embeds text using Gemini's embedding model
- Answers queries using Gemini-1.5-Flash
- Returns **JSON-formatted**, explainable responses with article title, URL, and image
- Exposes the system via a user-friendly Streamlit interface

---

## Usage
1. Clone the repository
```bash
git clone https://github.com/ostik24/RAG-system
cd RAG-system
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run app.py
```
4. Inside the app, you should set your Gemini API Key to proceed
You’ll be prompted on first run. The key is securely stored in session state.

---

## Architecture

```text
article_parser.py  →  Extracts articles, titles, and images from The Batch
chunker.py         →  Splits content into semantically meaningful chunks
embedder.py        →  Embeds and stores chunks in Chroma vector DB
rag_chain.py       →  Retrieves + reasons using Gemini and LangChain
app.py             →  Streamlit UI for asking and displaying answers
```

---

## Tools & Technologies
| LLM Embeddings - `models/embedding-001` - Gemini-compatible vector representations
| Vector DB - `Chroma + langchain-chroma` - Lightweight, persistent local DB
| Question Answering - `models/gemini-1.5-flash` - Fast, accurate, JSON-structured outputs
| UI - `Streamlit` - Simple and fast for demo/testing
| Parsing - `BeautifulSoup + requests` - Robust HTML parsing
| Chunking - `RecursiveCharacterTextSplitter` - Ensures context continuity

---

## Example Chunk Format

```json
{
  "id": "chunk_42",
  "content": "Wing launched its consumer drone delivery...",
  "metadata": {
    "article_title": "Drones Go Commercial",
    "issue_title": "AI and Drones",
    "issue_url": "https://www.deeplearning.ai/the-batch/issue-76/",
    "image_url": "https://.../drone.gif"
  }
}

---

## Evaluation Summary (RAGAS)
Faithfulness - 0.56 - Some hallucination detected
Answer Relevancy - 0.88 - Very relevant answers
Context Precision - 1.00 - Perfect use of retrieved docs
Context Recall - 1.00 - All needed context retrieved
Answer Correctness - 0.49 - Mixed factual accuracy

---

## Demo video
Watch the [video](https://drive.google.com/file/d/14DgQkKIFMBZIeDk6hfSNwmUYWE3VFeOF/view?usp=sharing)

---
