import os
os.system('pip install -q dnspython python-whois bs4 requests pandas pyOpenSSL')


import dns.resolver, dns.rdatatype
import requests
from bs4 import BeautifulSoup
from collections import Counter
import whois
from datetime import datetime
import time
import csv
import ssl
import socket
from urllib.request import urlparse
import OpenSSL.crypto
import pandas as pd
import random


def generate_user_agent() -> str:
    a = random.randint(63, 89)
    b = random.randint(1, 3200)
    c = random.randint(1, 140)
    user_agent = f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{a}.0.{b}.{c} Safari/537.3'
    return user_agent


headers = {
    'User-Agent': generate_user_agent()
}


def count_domain_occurrences(soup: BeautifulSoup, domain: str) -> int:
    """
    Returns the number of occurrences of the domain in the website's page source.
    """
    try:
        domain_count = soup.prettify().count(domain)
        return domain_count
    except Exception as e:
        print(f"count_domain_occurrences: {str(e)}")
        return 0


def get_certificate_info(url: str) -> tuple[str, int]:
    """
    Returns the issuer and age of the certificate if found. None, None otherwise
    """

    try:
        if not url.startswith("https://"):
            raise ValueError("URL must use HTTPS protocol")

        hostname = url.split("https://")[1].split("/")[0]
        ip_addresses = socket.getaddrinfo(hostname, 443)
        ip_address = ip_addresses[0][4][0]

        context = ssl.create_default_context()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ssl_conn = context.wrap_socket(sock, server_hostname=hostname)
        ssl_conn.connect((ip_address, 443))
        cert = ssl_conn.getpeercert()

        if 'notAfter' not in cert:
            raise ValueError("Certificate information not found")

        issuer = cert['issuer'][0][0][1]
        not_after = cert['notAfter']

        not_after_date = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
        certificate_age = (datetime.now() - not_after_date).days

        return issuer, certificate_age

    except Exception as e:
        print(f"get_certificate_info error: {str(e)}")

    return None, None


def check_sfh(soup: BeautifulSoup, domain: str) -> float:
    """
    Return 1 if SFH is "about: blank" or is empty, 0.5 if SFH refers to a different domain, and 0 otherwise
    """
    try:
        form = soup.find('form', {'method': 'post'})

        if not form:
            return 0

        sfh = form.get('action')

        if not sfh or sfh == 'about:blank':
            return 1
        
        sfh_domain = urlparse(sfh).netloc

        if sfh_domain != domain:
            return 0.5
    except Exception as e:
        print(f"check_sfh: {str(e)}")
        pass

    return 0


def age_of_domain(w: whois.WhoisEntry) -> int:
    """
    Returns the age of domain in days, None if error
    """
    try:
        creation_date = w.creation_date
        
        if creation_date is None:
            # Domain creation date is not available, try using updated_date as a fallback
            updated_date = w.updated_date
            if updated_date is None:
                return -1
            if type(updated_date) == list:
                creation_date = min(updated_date)
            else:
                creation_date = updated_date
        
        if type(creation_date) == list:
            creation_date = min(creation_date)
        
        num_days = (datetime.now() - creation_date).days
        
        return num_days
    except Exception as e:
        print('age_of_domain error: ' + str(e))
        return None


def use_iframe(soup: BeautifulSoup) -> int:
    """
    Returns 1 if iframe is present, 0 otherwise
    """
    iframes = soup.find_all('iframe')
    if len(iframes) > 0:
        return 1
    
    return 0


def popup_window_has_text_field(soup: BeautifulSoup) -> int:
    """
    Returns 1 if a pop-up window with text field exists, 0 otherwise
    """
    popups = soup.find_all('div', {'class': 'popup'})
    for popup in popups:
        if popup.find('input', {'type': 'text'}):
            return 1
    
    return 0


def abnormal_url(url: str, w: whois.WhoisEntry) -> int:
    """
    Returns 1 if the hostname is not in the URL, 0 otherwise.
    """
    host_name = w.domain.split('.')[0]
    if host_name not in url:
        return 1
    else:
        return 0


def dns_record(domain: str) -> tuple[int, int, int]:
    """
    Returns TTL, IP address count and TXT record presence in a tuple of integers.
    Returns None, None, None if dns record not found.
    """
    try:
        answers = dns.resolver.resolve(domain)
        TTL = answers.rrset.ttl
        IP_addresses = len(answers)
        TXT_records = any(answer.rdtype == dns.rdatatype.TXT for answer in answers)
        TXT_records = 1 if TXT_records else 0

        return TTL, IP_addresses, TXT_records
    except dns.resolver.NXDOMAIN:
        return None, None, None
    except Exception as e:
        print(f"dns_record error: {str(e)}")
        return None, None, None



def not_indexed_by_google(url: str) -> int:
    """
    Returns 1 if not indexed by Google, 0 if indexed, -1 if error
    """
    response = make_request(url, headers, timeout=10, retries=3)
    if response is None:
        return -1

    if "did not match any documents" in response.text:
        return 1
    else:
        return 0


def right_click_disabled(soup: BeautifulSoup) -> int:
    """
    Returns 1 if right click is disabled, 0 otherwise.
    """
    for script in soup.find_all('script'):
        if 'event.button==2' in script.text:
            return 1
    return 0


def mouseover_changes(soup: BeautifulSoup) -> int:
    """
    Returns 1 if the mouseover event changes the status bar, 0 otherwise
    """
    onMouseOver_elements = soup.find_all(onmouseover=True)
    for element in onMouseOver_elements:
        if "window.status" in str(element):
            return 1
    return 0


def redirects(response: requests.Response) -> int:
    """
    Returns the number of redirects
    """
    return len(response.history)


def meta_script_link_percentage(soup: BeautifulSoup) -> tuple[float, float, float]:
    """
    Returns the percentage of meta, script, and link tags that have a link
    """
    meta_tags = soup.find_all('meta')
    script_tags = soup.find_all('script')
    link_tags = soup.find_all('link')

    meta_links = sum([1 for tag in meta_tags if tag.has_attr('href')])
    script_links = sum([1 for tag in script_tags if tag.has_attr('src')])
    link_links = sum([1 for tag in link_tags if tag.has_attr('href')])

    total_links = meta_links + script_links + link_links
    if total_links == 0:
        return 0, 0, 0
    meta_percentage = (meta_links / total_links)
    script_percentage = (script_links / total_links)
    link_percentage = (link_links / total_links)

    return meta_percentage, script_percentage, link_percentage


def url_anchor_percentage(soup: BeautifulSoup) -> float:
    """
    Returns the percentage of anchor links on the page with different domain names,
    excluding anchor links with JavaScript or invalid URLs.
    """
    total_links = 0
    anchor_links = 0

    first_a_tag = soup.find('a')
    if first_a_tag is None:
        return 0

    domain = urlparse(first_a_tag.get('href')).netloc
    if not domain:
        return 0

    for a_tag in soup.find_all('a'):
        href = a_tag.get('href')
        if href:
            if href.startswith('javascript:') or href.startswith('#'):
                continue

            link_domain = urlparse(href).netloc
            if link_domain and link_domain != domain:
                anchor_links += 1
            total_links += 1

    if total_links == 0:
        return 0

    return anchor_links / total_links


def request_url_percentage(soup: BeautifulSoup, domain: str) -> float:
    """
    Returns the percentage of external domains in the URL
    """ 
    links = [link.get('href') for link in soup.find_all('a')]
    images = [img.get('src') for img in soup.find_all('img')]
    videos = [video.get('src') for video in soup.find_all('video')]
    sounds = [sound.get('src') for sound in soup.find_all('audio')]
    external_links = []
    
    for link in links + images + videos + sounds:
        if link is None:
            continue
        parsed_domain = urlparse(link).netloc
        if parsed_domain != '' and parsed_domain != domain:
            external_links.append(link)
    
    external_domains = [urlparse(link).netloc for link in external_links]
    domain_counts = Counter(external_domains)
    
    total_links = len(external_domains)
    if total_links == 0:
        return 1
    external_links_count = domain_counts[domain]
    
    return (external_links_count / total_links)


def has_suspicious_port(domain: str) -> int:
    """
    Returns 1 if any of the ports are of the preferred status, 0 otherwise.
    """
    preferred_ports = {
        21: "Close",
        22: "Close",
        23: "Close",
        80: "Open",
        443: "Open",
        445: "Close",
        1433: "Close",
        1521: "Close",
        3306: "Close",
        3389: "Close"
    }
    for port, status in preferred_ports.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex((domain, port))
            if result == 0:
                if status == "Open":
                    return 1
                else:
                    return 0
        except:
            pass
        
    return 0


def external_favicons(soup: BeautifulSoup, domain: str) -> int:
    """
    Returns the number of favicons loaded from external domains.
    """
    favicon_links = soup.find_all('link', {'rel': 'icon'})
    external_favicons = 0

    for link in favicon_links:
        href = link.get('href')

        if href:
            href_domain = urlparse(href).netloc

            if href_domain != domain:
                external_favicons += 1

    return external_favicons


def domain_registeration_length(w: whois.WhoisEntry) -> int:
    """"
    Returns the number of days since the domain was registered, None if error
    """
    try:
        domain = w.domain_name
        expiration_date = w.expiration_date
        if type(expiration_date) == list:
            expiration_date = expiration_date[0]
        if expiration_date is not None:
            time_to_expire = (expiration_date - datetime.now()).days
            return time_to_expire
        else:
            return 0
    except Exception as e:
        print('domain_registeration_length error: ' + str(e))
        return None


def check_email_submission(soup: BeautifulSoup) -> int:
    """
    Returns 1 if "mail()" or "mailto:" is used, 0 otherwise.
    """
    try:
        forms = soup.find_all('form')
        for form in forms:
            if 'mail(' in str(form) or 'mailto:' in str(form):
                return 1
        return 0
    except:
        return 0
    

def make_request(url: str, headers: dict, timeout: int, retries: int) -> requests.Response:
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            retry_delay = 2**i
            print(f'\033[34mRequestException for {url}: {e}. Retrying in {retry_delay} seconds...\033[0m')
            time.sleep(retry_delay)
        except Exception as e:
            print(f'\033[31mError making request for {url}: {e}\033[0m')
            return None
    print(f'\033[31mFailed to make request after {retries} retries.\033[0m')
    return None


def collect_data(url: str, is_malicious: bool):
    start_time = time.time()

    try:
        response = make_request(url, headers, timeout=10, retries=3)
        if response is None:
            return
        redirects_value = redirects(response)        
    except Exception as e:
        print(f'Error making request: {e}')
        return
    not_indexed_by_google_value = not_indexed_by_google(url)
    issuer, certificate_age = get_certificate_info(url)
    
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        email_submission_value = check_email_submission(soup)
        url_anchor_percentage_value = url_anchor_percentage(soup)
        meta_percentage, script_percentage, link_percentage = meta_script_link_percentage(soup)
        mouseover_changes_value = mouseover_changes(soup)
        right_click_disabled_value = right_click_disabled(soup)
        popup_window_has_text_field_value = popup_window_has_text_field(soup)
        use_iframe_value = use_iframe(soup)
    except Exception as e:
        print('soup error, double check your code: ' + str(e))
        return
    
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        has_suspicious_port_value = has_suspicious_port(domain)
        request_url_percentage_value = request_url_percentage(soup, domain)
        external_favicons_value = external_favicons(soup, domain)
        TTL, ip_address_count, TXT_record = dns_record(domain)
        check_sfh_value = check_sfh(soup, domain)
        count_domain_occurrences_value = count_domain_occurrences(soup, domain)
    except Exception as e:
        print('urlparse error, double check your code: ' + str(e))
        return
    
    try:
        w = whois.whois(domain)
        domain_registeration_length_value = domain_registeration_length(w)
        abnormal_url_value = abnormal_url(url, w)
        age_of_domain_value = age_of_domain(w)
    except Exception as e:
        print('whois error: ' + str(e))
        domain_registeration_length_value = None
        abnormal_url_value = None
        age_of_domain_value = None
    
    
    print(f"{url} took {time.time() - start_time} seconds to complete")

    row = [url, redirects_value, not_indexed_by_google_value, issuer, certificate_age, email_submission_value, request_url_percentage_value, url_anchor_percentage_value, meta_percentage, script_percentage, link_percentage, mouseover_changes_value, right_click_disabled_value, popup_window_has_text_field_value, use_iframe_value, has_suspicious_port_value, external_favicons_value, TTL, ip_address_count, TXT_record, check_sfh_value, count_domain_occurrences_value, domain_registeration_length_value, abnormal_url_value, age_of_domain_value, is_malicious]
    
    with open('phishing_detection_dataset.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)