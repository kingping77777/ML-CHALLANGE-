import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict
import rapidfuzz
from rapidfuzz import distance, process, fuzz

def compute_string_features(df: pd.DataFrame, col1: str, col2: str, prefix: str) -> pd.DataFrame:
    """
    Computes comprehensive string similarity features between col1 and col2.
    """
    print(f"Computing string features for {col1} and {col2} with prefix '{prefix}'...")
    
    s1_series = df[col1].fillna('').astype(str).str.lower().str.strip()
    s2_series = df[col2].fillna('').astype(str).str.lower().str.strip()
    
    # Exact matches
    df[f'{prefix}_exact_match'] = (s1_series == s2_series).astype(np.int8)
    
    s1_compact = s1_series.str.replace(r'\s+', '', regex=True)
    s2_compact = s2_series.str.replace(r'\s+', '', regex=True)
    df[f'{prefix}_compact_exact_match'] = ((s1_compact == s2_compact) & (s1_compact != '')).astype(np.int8)
    
    # Length metrics
    len1 = s1_series.str.len().astype(np.float32)
    len2 = s2_series.str.len().astype(np.float32)
    max_len = np.maximum(len1, len2)
    min_len = np.minimum(len1, len2)
    
    df[f'{prefix}_length_ratio'] = np.where(max_len > 0, min_len / max_len, 0.0).astype(np.float32)
    df[f'{prefix}_length_difference'] = np.abs(len1 - len2).astype(np.float32)
    
    # Rapidfuzz Levenshtein & Jaro-Winkler & Token Sort
    n_rows = len(df)
    s1_list = s1_series.tolist()
    s2_list = s2_series.tolist()
    
    edit_sim = np.zeros(n_rows, dtype=np.float32)
    jw_sim = np.zeros(n_rows, dtype=np.float32)
    sorted_token_sim = np.zeros(n_rows, dtype=np.float32)
    jaccard_arr = np.zeros(n_rows, dtype=np.float32)
    overlap_s1_arr = np.zeros(n_rows, dtype=np.float32)
    overlap_cand_arr = np.zeros(n_rows, dtype=np.float32)
    ngram_jaccard_arr = np.zeros(n_rows, dtype=np.float32)
    ngram_overlap_arr = np.zeros(n_rows, dtype=np.float32)
    
    for i in range(n_rows):
        v1 = s1_list[i]
        v2 = s2_list[i]
        if not v1 or not v2:
            continue
            
        # Edit similarity & Jaro Winkler
        edit_sim[i] = distance.Levenshtein.normalized_similarity(v1, v2)
        jw_sim[i] = distance.JaroWinkler.similarity(v1, v2)
        sorted_token_sim[i] = fuzz.token_sort_ratio(v1, v2) / 100.0
        
        # Token metrics
        t1 = set(v1.split())
        t2 = set(v2.split())
        if t1 and t2:
            inter = len(t1 & t2)
            union = len(t1 | t2)
            jaccard_arr[i] = inter / union
            overlap_s1_arr[i] = inter / len(t1)
            overlap_cand_arr[i] = inter / len(t2)
            
        # 4-gram metrics
        g1 = set([v1[j:j+4] for j in range(len(v1)-3)]) if len(v1) >= 4 else set([v1])
        g2 = set([v2[j:j+4] for j in range(len(v2)-3)]) if len(v2) >= 4 else set([v2])
        if g1 and g2:
            g_inter = len(g1 & g2)
            g_union = len(g1 | g2)
            ngram_jaccard_arr[i] = g_inter / g_union
            ngram_overlap_arr[i] = g_inter / min(len(g1), len(g2))

    df[f'{prefix}_token_jaccard'] = jaccard_arr
    df[f'{prefix}_token_overlap_s1'] = overlap_s1_arr
    df[f'{prefix}_token_overlap_candidate'] = overlap_cand_arr
    df[f'{prefix}_edit_similarity'] = edit_sim
    if prefix == 'name':
        df[f'{prefix}_jaro_winkler'] = jw_sim
        df[f'{prefix}_sorted_token_similarity'] = sorted_token_sim
    df[f'{prefix}_char_4gram_jaccard'] = ngram_jaccard_arr
    df[f'{prefix}_char_4gram_overlap'] = ngram_overlap_arr
    
    # TF-IDF Cosine placeholder / simple character overlap ratio for tfidf representation
    df[f'{prefix}_tfidf_cosine'] = ngram_jaccard_arr.astype(np.float32)
    
    return df


def compute_structured_features(df: pd.DataFrame, s1_prefix: str, cand_prefix: str) -> pd.DataFrame:
    """
    Computes structured address and metadata matching features.
    """
    print("Computing structured address features...")
    
    def match_feature(c1, c2):
        s1_val = df[c1].fillna('').astype(str).str.lower().str.strip()
        s2_val = df[c2].fillna('').astype(str).str.lower().str.strip()
        both_avail = ((s1_val != '') & (s2_val != '')).astype(np.int8)
        match = ((s1_val == s2_val) & (both_avail == 1)).astype(np.int8)
        return both_avail, match

    # Postal code
    if f'{s1_prefix}_postal' in df.columns and f'{cand_prefix}_postal' in df.columns:
        both_postal, match_postal = match_feature(f'{s1_prefix}_postal', f'{cand_prefix}_postal')
        df['postal_available_both'] = both_postal
        df['postal_match'] = match_postal
    else:
        df['postal_available_both'] = np.int8(0)
        df['postal_match'] = np.int8(0)
    
    # House number
    if f'{s1_prefix}_house_number' in df.columns and f'{cand_prefix}_house_number' in df.columns:
        both_hn, match_hn = match_feature(f'{s1_prefix}_house_number', f'{cand_prefix}_house_number')
        df['house_number_available_both'] = both_hn
        df['house_number_match'] = match_hn
    else:
        df['house_number_available_both'] = np.int8(0)
        df['house_number_match'] = np.int8(0)
    
    # City
    if f'{s1_prefix}_city' in df.columns and f'{cand_prefix}_city' in df.columns:
        _, df['city_match'] = match_feature(f'{s1_prefix}_city', f'{cand_prefix}_city')
    else:
        df['city_match'] = np.int8(0)
        
    # Locality
    if f'{s1_prefix}_locality' in df.columns and f'{cand_prefix}_locality' in df.columns:
        _, df['locality_match'] = match_feature(f'{s1_prefix}_locality', f'{cand_prefix}_locality')
    else:
        df['locality_match'] = np.int8(0)

    # State
    if f'{s1_prefix}_state' in df.columns and f'{cand_prefix}_state' in df.columns:
        _, df['state_match'] = match_feature(f'{s1_prefix}_state', f'{cand_prefix}_state')
    else:
        df['state_match'] = np.int8(0)
        
    return df


def compute_multilingual_features(df: pd.DataFrame, s1_col: str, cand_col: str) -> pd.DataFrame:
    """
    Computes multilingual and script specific metadata features.
    """
    print("Computing multilingual features...")
    
    s1_text = df[s1_col].fillna('').astype(str)
    cand_text = df[cand_col].fillna('').astype(str)
    
    # Non-ASCII detection
    df['name_has_non_ascii_s1'] = s1_text.str.contains(r'[^\x00-\x7F]', regex=True).astype(np.int8)
    df['name_has_non_ascii_candidate'] = cand_text.str.contains(r'[^\x00-\x7F]', regex=True).astype(np.int8)
    
    # Devanagari detection (\u0900-\u097F)
    df['name_has_devanagari_s1'] = s1_text.str.contains('[\u0900-\u097F]', regex=True).astype(np.int8)
    df['name_has_devanagari_candidate'] = cand_text.str.contains('[\u0900-\u097F]', regex=True).astype(np.int8)
    
    # Same script indicator
    both_ascii = ((df['name_has_non_ascii_s1'] == 0) & (df['name_has_non_ascii_candidate'] == 0)).astype(np.int8)
    both_devanagari = ((df['name_has_devanagari_s1'] == 1) & (df['name_has_devanagari_candidate'] == 1)).astype(np.int8)
    df['both_name_same_script'] = np.maximum(both_ascii, both_devanagari).astype(np.int8)
    
    return df


def compute_metadata_features(df: pd.DataFrame, s1_prefix: str, cand_prefix: str) -> pd.DataFrame:
    """
    Computes metadata features such as country match and source encoding.
    """
    print("Computing metadata features...")
    
    # Country match
    c1 = f'{s1_prefix}_country' if f'{s1_prefix}_country' in df.columns else 's1_country'
    c2 = f'{cand_prefix}_country' if f'{cand_prefix}_country' in df.columns else 'cand_country'
    
    if c1 in df.columns and c2 in df.columns:
        s1_country = df[c1].fillna('').astype(str).str.upper().str.strip()
        cand_country = df[c2].fillna('').astype(str).str.upper().str.strip()
        both_avail = ((s1_country != '') & (cand_country != '')).astype(np.int8)
        df['country_match'] = ((s1_country == cand_country) & (both_avail == 1)).astype(np.int8)
    else:
        df['country_match'] = np.int8(0)
        
    # Source encoding
    if 'candidate_source' in df.columns:
        df['candidate_source_S2'] = (df['candidate_source'] == 'S2').astype(np.int8)
        df['candidate_source_S3'] = (df['candidate_source'] == 'S3').astype(np.int8)
    else:
        df['candidate_source_S2'] = np.int8(0)
        df['candidate_source_S3'] = np.int8(0)
        
    return df


def compute_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes explicit missingness indicators.
    """
    print("Computing missingness features...")
    
    df['s1_name_missing'] = df['s1_name'].isna() | (df['s1_name'].astype(str).str.strip() == '')
    df['s1_name_missing'] = df['s1_name_missing'].astype(np.int8)
    
    df['candidate_name_missing'] = df['cand_name'].isna() | (df['cand_name'].astype(str).str.strip() == '')
    df['candidate_name_missing'] = df['candidate_name_missing'].astype(np.int8)
    
    df['s1_address_missing'] = df['s1_address'].isna() | (df['s1_address'].astype(str).str.strip() == '')
    df['s1_address_missing'] = df['s1_address_missing'].astype(np.int8)
    
    df['candidate_address_missing'] = df['cand_address'].isna() | (df['cand_address'].astype(str).str.strip() == '')
    df['candidate_address_missing'] = df['candidate_address_missing'].astype(np.int8)
    
    s1_postal_col = 's1_postal' if 's1_postal' in df.columns else 's1_address'
    cand_postal_col = 'cand_postal' if 'cand_postal' in df.columns else 'cand_address'
    df['s1_postal_missing'] = df[s1_postal_col].isna() | (df[s1_postal_col].astype(str).str.strip() == '')
    df['s1_postal_missing'] = df['s1_postal_missing'].astype(np.int8)
    
    df['candidate_postal_missing'] = df[cand_postal_col].isna() | (df[cand_postal_col].astype(str).str.strip() == '')
    df['candidate_postal_missing'] = df['candidate_postal_missing'].astype(np.int8)
    
    return df


def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the generation of all feature groups:
    1. Business Name
    2. Address
    3. Structured Address
    4. Multilingual / Script
    5. Dense Semantic Similarity
    6. Metadata
    7. Missingness
    """
    df = df.copy()
    
    # 1. Business Name
    df = compute_string_features(df, 's1_name', 'cand_name', 'name')
    
    # 2. Address
    df = compute_string_features(df, 's1_address', 'cand_address', 'address')
    
    # 3. Structured Address
    df = compute_structured_features(df, 's1', 'cand')
    
    # 4. Multilingual / Script
    df = compute_multilingual_features(df, 's1_name', 'cand_name')
    
    # 5. Dense Semantic Similarity
    if 'dense_cosine_similarity' not in df.columns:
        df['dense_cosine_similarity'] = np.float32(0.0)
    df['dense_name_cosine'] = np.float32(0.0)
    df['dense_address_cosine'] = np.float32(0.0)
    
    # 6. Metadata
    df = compute_metadata_features(df, 's1', 'cand')
    
    # 7. Missingness
    df = compute_missingness_features(df)
    
    return df
