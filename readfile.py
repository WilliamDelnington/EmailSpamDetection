import re
from bs4 import BeautifulSoup

def extract_email(file):
    with open(file, "r") as f:
        content = f.read()

    content = content.lower()

    patterns = r'\<html.*?\>(.*?)\<\/html\>'
    match = re.search(patterns, content, re.DOTALL)
    if match:
        html_content = match.group(1)

        bs = BeautifulSoup(html_content, "html.parser")

        html_content = bs.get_text().strip()

        html_content = re.sub(r'\s+', ' ', html_content)

        # print(html_content[:700])
        return html_content
    else:
        print(f"No match found in file {file}")
        return None
    
# print(extract_email("E:/Python Tests/AI/EmailSpamDetection\Datasets/spam_2/spam_2/01267.6e9de9ba20f59c26028ebdb4550685fb"))
print(extract_email("E:/Python Tests/AI/EmailSpamDetection/Datasets/spam_2/spam_2/01320.9d28c111c72720b5cd64a025591dbce5"))