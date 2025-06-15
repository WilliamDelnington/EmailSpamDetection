from urllib.parse import urlparse, urlunparse, parse_qs
import pandas as pd
import re
def cleanup_url_dataset(df:pd.DataFrame, url_column='url', inplace=False):
    """
    Clean up a URL dataset by removing duplicates, handling missing values,
    normalizing URLs, and performing various cleanup operations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing URLs
    url_column : str
        Name of the column containing URLs (default: 'url')
    inplace : bool
        Whether to modify the original DataFrame (default: False)
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    dict
        Cleanup statistics
    """
    
    if not inplace:
        df = df.copy()
    
    # Initialize statistics
    stats = {
        'original_count': len(df),
        'missing_values_removed': 0,
        'invalid_urls_removed': 0,
        'duplicates_removed': 0,
        'final_count': 0
    }
    
    print("Starting URL dataset cleanup...")
    
    # 1. Handle missing values
    initial_count = len(df)
    df = df.dropna(subset=[url_column])
    stats['missing_values_removed'] = initial_count - len(df)
    print(f"Removed {stats['missing_values_removed']} rows with missing URLs")
    
    # 2. Remove empty strings and whitespace-only URLs
    df = df[df[url_column].str.strip() != '']
    
    # 3. Basic URL cleaning and normalization
    def clean_url(url):
        if pd.isna(url):
            return None
            
        # Convert to string and strip whitespace
        url = str(url).strip()
        
        # Convert to lowercase
        url = url.lower()
        
        # Add http:// if no scheme is present
        if not url.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
            url = 'http://' + url
        
        try:
            # Parse and reconstruct URL for normalization
            parsed = urlparse(url)
            
            # Remove default ports
            netloc = parsed.netloc
            if netloc.endswith(':80') and parsed.scheme == 'http':
                netloc = netloc[:-3]
            elif netloc.endswith(':443') and parsed.scheme == 'https':
                netloc = netloc[:-4]
            
            # Remove trailing slash from path if it's just '/'
            path = parsed.path
            if path == '/':
                path = ''
            elif path.endswith('/') and len(path) > 1:
                path = path.rstrip('/')
            
            # Sort query parameters for consistency
            query = parsed.query
            if query:
                query_params = sorted(query.split('&'))
                query = '&'.join(query_params)
            
            # Reconstruct URL
            cleaned_parsed = parsed._replace(
                netloc=netloc,
                path=path,
                query=query,
                fragment=''  # Remove fragments for consistency
            )
            
            return urlunparse(cleaned_parsed)
            
        except Exception:
            return None
    
    # Apply URL cleaning
    df[url_column] = df[url_column].apply(clean_url)
    
    # 4. Remove invalid URLs (None values after cleaning)
    before_invalid = len(df)
    df = df.dropna(subset=[url_column])
    stats['invalid_urls_removed'] = before_invalid - len(df)
    print(f"Removed {stats['invalid_urls_removed']} invalid URLs")
    
    # 5. Additional URL validation
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    df = df[df[url_column].apply(is_valid_url)]
    
    # 6. Remove duplicates
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=[url_column], keep='first')
    stats['duplicates_removed'] = before_duplicates - len(df)
    print(f"Removed {stats['duplicates_removed']} duplicate URLs")
    
    # 7. Optional: Remove common non-content URLs (uncomment if needed)
    # common_excludes = [
    #     r'.*\.(css|js|ico|png|jpg|jpeg|gif|svg|pdf|zip|exe)$',
    #     r'.*/(wp-admin|admin|login|logout).*',
    #     r'.*\?.*utm_.*',  # Remove URLs with UTM parameters
    # ]
    # 
    # for pattern in common_excludes:
    #     df = df[~df[url_column].str.contains(pattern, regex=True, na=False)]
    
    # Final statistics
    stats['final_count'] = len(df)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"\nCleanup completed!")
    print(f"Original count: {stats['original_count']}")
    print(f"Final count: {stats['final_count']}")
    print(f"Total removed: {stats['original_count'] - stats['final_count']}")
    
    return df, stats

def extract_url_features(df, url_column="url"):
    def get_features(url):
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            hostname = parsed.hostname if parsed.hostname else ""
            path = parsed.path if parsed.path else ""

            return pd.Series({
                "url_length": len(url),
                "hostname_length": len(hostname),
                "path_length": len(path),
                "num_dots": url.count('.'),
                "num_slashes": url.count('/'),
                "num_digits": len(re.findall(r'\d', url)),
                "num_special_chars": len(re.findall(r'[^\w\s]', url)),
                "has_https": int(parsed.scheme == 'https'),
                "has_ip": int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname))),
                "has_port": int(bool(parsed.port)),
                "has_at_symbol": int('@' in url),
                "has_query": int(bool(parsed.query)),
                "num_params": len(query_params),
                "tld": hostname.split('.')[-1] if '.' in hostname else '',
                "domain_in_path": int(hostname in path),
            })
        except Exception:
            # In case of any parsing error, return defaults
            return pd.Series({
                "url_length": 0,
                "hostname_length": 0,
                "path_length": 0,
                "num_dots": 0,
                "num_slashes": 0,
                "num_digits": 0,
                "num_special_chars": 0,
                "has_https": 0,
                "has_ip": 0,
                "has_port": 0,
                "has_at_symbol": 0,
                "has_query": 0,
                "num_params": 0,
                "tld": '',
                "domain_in_path": 0,
            })

    # Apply the feature extraction row-wise
    features_df = df[url_column].apply(get_features)
    return pd.concat([df, features_df], axis=1)