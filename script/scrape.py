from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber
import re

# List of URLs to scrape
urls = [
    'https://perfectdailygrind.com/2019/11/how-packaging-material-impacts-green-coffee-quality-over-time/',
    'https://dfreight.org/blog/shipping-coffee/',
    'https://packhelp.com/coffee-packaging-keeping-coffee-fresh/?srsltid=AfmBOorY5CHh9SsAKIqBhnbWiMB4N1oTfIGSlMLwhjouUur7p_EXmKcT',
    'https://www.sttark.com/blog/ultimate-guide-to-coffee-packaging?srsltid=AfmBOoooQNz0YUO0r8l6-fSAxjZIbL6rLJr2Ur0rrHOEvegkhwCxI5SK',
    'https://www.levapack.com/what-types-of-coffee-packaging-do-you-need/',
    'https://bellwethercoffee.com/blog/the-roasters-guide-to-coffee-packaging',
    'https://vikingmasek.com/packaging-machine-resources/packaging-machine-blog/4-factors-to-consider-when-selecting-coffee-packaging',
    'https://www.freightcenter.com/shipping/ultimate-guide-to-shipping-coffee/#:~:text=From%20Export%20to%20Consumption',
    'https://www.mcmc.gov.my/en/sectors/postal-courier/policies',
    'https://www.mcmc.gov.my/en/sectors/postal-courier'
]

categories = [
    'Packaging Beans',  
    'Packaging Beans',   
    'Packaging Beans', 
    'Packaging Beans',  
    'Packaging Beans',   
    'Packaging Beans',
    'Packaging Beans',  
    'Shipping Beans',
    'MCMC Regulation',
    'MCMC Regulation'   
]

selectors = [
    ('div', 'single-content'),  
    ('div', 'wp-site-blocks'),  
    ('div', 'content-area'),
    ('div', 'blog-post'),  
    ('div', 'elementor-widget-container'),  
    ('div', 'sqs-block html-block sqs-block-html'),
    ('div', 'col-sm-8 col-content blog-content'),
    ('div', 's-delivery__container'),
    ('div', 'accTxtArea'),
    ('div', 'content')
]

def clean_text(text):
    """
    Cleans the text by removing unwanted characters and irrelevant information.
    """
    cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    cleaned_text = re.sub(r'[^\w\s,.!?]', '', cleaned_text)  # Remove non-alphabetic characters except common punctuation
    cleaned_text = cleaned_text.strip()  # Trim any leading/trailing whitespace
    return cleaned_text

data = {
    'Category': [],
    'Scraped Content': []
}

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

for url, category, selector in zip(urls, categories, selectors):
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    tag, class_name = selector[:2]  
    kwargs = selector[2] if len(selector) > 2 else {}  
    article_body = soup.find(tag, class_=class_name, **kwargs)  
    
    if article_body:
        article_text = article_body.get_text(separator='\n', strip=True)
        cleaned_article_text = clean_text(article_text)
        print(f"Scraped content from {url}")
    else:
        cleaned_article_text = "Could not find the article content"
        print(f"Could not find the article content at {url}")
    
    data['Category'].append(category)
    data['Scraped Content'].append(cleaned_article_text)

driver.quit()

pdf_file_path = r'C:\Users\Enduser\Downloads\LG_for-non-universal-service-licence-application_latest.pdf' #code was done locally

with pdfplumber.open(pdf_file_path) as pdf:
    pdf_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            pdf_text += clean_text(text) + "\n"  

data['Category'].append('MCMC regulation')
data['Scraped Content'].append(pdf_text)

df = pd.DataFrame(data)

csv_file_path = r'C:\Users\Enduser\OneDrive - Asia Pacific University\uni\Y3S1\scraped_articles.csv' #code was done locally

df.to_csv(csv_file_path, index=False, encoding='utf-8')

print(f"Data saved to {csv_file_path}")
