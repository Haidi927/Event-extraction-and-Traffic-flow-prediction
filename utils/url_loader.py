import requests
from bs4 import BeautifulSoup

def extract_article_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return '\n'.join(p.text for p in paragraphs if p.text.strip())
