import pandas as pd
import numpy as np
import re
import unicodedata

def clean_text(text):
    if pd.isna(text):
        return ""
    # Unicode normalize (NFKC handles composed/decomposed characters robustly)
    text = unicodedata.normalize('NFKC', str(text))
    return text

def normalize_country(country_str):
    if pd.isna(country_str):
        return ""
    return clean_text(country_str).strip().lower()

def detect_scripts(text):
    if pd.isna(text):
        return False, False
    text = str(text)
    has_non_ascii = len(text) != len(text.encode('utf-8', 'ignore'))
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    return has_non_ascii, has_devanagari

def normalize_business_name(name):
    """
    Normalizes business names by converting to lowercase, handling ampersands, 
    stripping punctuation (while preserving Unicode letters), and normalizing legal suffixes.
    """
    if pd.isna(name):
        return ""
    
    # 1. Unicode & Case
    name = clean_text(name).lower()
    
    # 2. '&' to 'and'
    name = name.replace('&', ' and ')
    
    # 3. Punctuation removal (preserving words and Unicode characters like Devanagari)
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # 4. Whitespace collapse
    name = re.sub(r'\s+', ' ', name).strip()
    
    # 5. Common Legal Suffixes Normalization
    name = re.sub(r'\bpvt\b', 'private', name)
    name = re.sub(r'\bltd\b', 'limited', name)
    name = re.sub(r'\bllc\b', 'l l c', name) 
    name = re.sub(r'\bl l c\b', 'llc', name)
    name = re.sub(r'\binc\b', 'incorporated', name)
    name = re.sub(r'\bcorp\b', 'corporation', name)
    name = re.sub(r'\bco\b', 'company', name)
    
    # Final whitespace cleanup
    return re.sub(r'\s+', ' ', name).strip()

def compact_name(norm_name):
    if not norm_name:
        return ""
    return re.sub(r'\s+', '', norm_name)

def tokenize_name(norm_name):
    if not norm_name:
        return []
    return norm_name.split()

def normalize_address(address):
    """
    Normalizes addresses by converting to lowercase, removing non-word punctuation,
    and handling common street abbreviations.
    """
    if pd.isna(address):
        return ""
    
    addr = clean_text(address).lower()
    
    # Punctuation to space
    addr = re.sub(r'[^\w\s]', ' ', addr)
    
    # Abbreviations
    addr = re.sub(r'\brd\b', 'road', addr)
    addr = re.sub(r'\bst\b', 'street', addr)
    addr = re.sub(r'\bave\b', 'avenue', addr)
    addr = re.sub(r'\bblvd\b', 'boulevard', addr)
    addr = re.sub(r'\bdr\b', 'drive', addr)
    addr = re.sub(r'\bln\b', 'lane', addr)
    
    return re.sub(r'\s+', ' ', addr).strip()

def extract_postal_code(address):
    """
    Extracts observed PIN/ZIP codes natively from the raw string.
    Works for Indian PIN (6 digits) and US ZIP (5 digits or 5+4).
    """
    if pd.isna(address):
        return None
        
    address = str(address)
    
    # Indian PIN code: 6 consecutive digits
    india_pin = re.findall(r'\b\d{6}\b', address)
    if india_pin:
        return india_pin[-1]
        
    # US ZIP+4: 5 digits, hyphen, 4 digits
    us_zip4 = re.findall(r'\b\d{5}-\d{4}\b', address)
    if us_zip4:
        return us_zip4[-1]
        
    # US ZIP: 5 digits
    us_zip = re.findall(r'\b\d{5}\b', address)
    if us_zip:
        return us_zip[-1]
        
    return None

def preprocess_dataframe(df):
    """
    Applies all normalizations to a DataFrame, creating derived columns 
    while preserving the original data.
    """
    df_out = df.copy()
    
    # Country
    df_out['country_normalized'] = df_out['country'].apply(normalize_country)
    
    # Business Name
    df_out['business_name_normalized'] = df_out['business_name'].apply(normalize_business_name)
    df_out['business_name_compact'] = df_out['business_name_normalized'].apply(compact_name)
    df_out['business_name_tokenized'] = df_out['business_name_normalized'].apply(tokenize_name)
    
    # Unicode Script tracking
    scripts = df_out['business_name'].apply(detect_scripts)
    df_out['name_has_non_ascii'] = [s[0] for s in scripts]
    df_out['name_has_devanagari'] = [s[1] for s in scripts]
    
    # Address
    df_out['business_address_normalized'] = df_out['business_address'].apply(normalize_address)
    df_out['business_address_compact'] = df_out['business_address_normalized'].apply(compact_name)
    
    # Postal Code
    df_out['postal_code'] = df_out['business_address'].apply(extract_postal_code)
    df_out['has_postal_code'] = df_out['postal_code'].notna().astype(int)
    
    return df_out
