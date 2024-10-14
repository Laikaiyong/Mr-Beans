from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd  

# Setup WebDriver using webdriver-manager to handle the ChromeDriver installation
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# Define the URL to scrape
url = 'https://perfectdailygrind.com/2019/11/how-packaging-material-impacts-green-coffee-quality-over-time/'
driver.get(url)

# Get the page source after the page is fully loaded
html = driver.page_source

# Close the browser
driver.quit()

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# Find the article body (using the class name from the HTML structure)
article_body = soup.find('div', class_='single-content')

# Initialize an empty string for article text
article_text = ""

# Extract the text from the article
if article_body:
    article_text = article_body.get_text(separator='\n', strip=True)
    print(article_text)
else:
    print("Could not find the article content")

# Define the CSV file path
csv_file_path = r'C:\Users\Enduser\OneDrive - Asia Pacific University\uni\Y3S1\scraped_article.csv'

# Check if the CSV file exists
try:
    # Read the existing CSV file
    df = pd.read_csv(csv_file_path)

    # Append the new article text to the 'packaging beans' column
    df = df.append({'packaging beans': article_text}, ignore_index=True)

except FileNotFoundError:
    # If the file doesn't exist, create a new DataFrame
    df = pd.DataFrame({'packaging beans': [article_text]})

# Save the DataFrame back to the CSV file
df.to_csv(csv_file_path, index=False, encoding='utf-8')
