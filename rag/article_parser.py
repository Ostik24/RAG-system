"""
Parser for the DeepLearning.AI The Batch newsletter
"""

import requests
from bs4 import BeautifulSoup
import json
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                   (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

def extract_articles_with_images(soup):
    """
    Extract articles from the given issue.
    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the HTML content.
    Returns:
        list: A list of dictionaries, each containing the title, image URL, and content of an article.
    """
    articles = []
    h_tags = soup.find_all('h1') + soup.find_all('h2') + soup.find_all('h3')

    for i, h1 in enumerate(h_tags):
        title = h1.get_text(strip=True)
        lower_title = title.lower()
        if not title or 'news' in lower_title or 'issue' in lower_title or 'message' in lower_title or 'subscribe' in lower_title:
            continue
        image_url = None

        prev = h1.find_previous_sibling()
        while prev and prev.name != 'figure':
            prev = prev.find_previous_sibling()
        if prev and prev.name == 'figure':
            img_tag = prev.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_url = img_tag['src']

        article_parts = []
        next_h1 = h_tags[i + 1] if i + 1 < len(h_tags) else None
        current = h1.find_next_sibling()

        while current and current != next_h1:
            if current.name in ['p', 'ul']:
                article_parts.append(current.get_text(strip=True))
            current = current.find_next_sibling()

        if not article_parts:
            continue

        articles.append({
            "title": title,
            "image": image_url,
            "content": "\n".join(article_parts)
        })

    return articles

def extract_articles():
    """
    Extract articles from the given BeautifulSoup object.
    Args:
        None
    Returns:
        list: A list of dictionaries, each containing the title and content of an article.
    """
    json_with_issues = {}
    for i in range(1, 23):
        batch_url = f"https://www.deeplearning.ai/the-batch/page/{i}/"
        response = requests.get(batch_url)

        if response.status_code == 429:
            print(f"Page {i} returned status 429. Sleeping before retry...")
            time.sleep(25)
            response = requests.get(batch_url, headers=HEADERS)

        soup = BeautifulSoup(response.text, "html.parser")
        script_tag = soup.find("script", id="__NEXT_DATA__")

        data = json.loads(script_tag.string)

        posts = data["props"]["pageProps"]["posts"]

        for post in posts:
            title = post["title"]
            slug = post["slug"]
            issue_url = f"https://www.deeplearning.ai/the-batch/{slug}/"
            soup_url = requests.get(issue_url).text
            soup_urll = BeautifulSoup(soup_url, "html.parser")
            articles = extract_articles_with_images(soup_urll)
            json_with_issues[slug] = {
                "title": title,
                "url": issue_url,
                "articles": articles
            }

    return json_with_issues
