import collections
import math
from array import array
import pandas as pd
import numpy as np

def get_4grams(text):
    if not text or len(text) < 4:
        return [text] if text else []
    return [text[i:i+4] for i in range(len(text)-3)]

class SparseBlocker:
    """
    A high-performance, memory-efficient Inverted Index for sparse blocking.
    Uses array('i') internally to store integer IDs instead of string objects,
    reducing memory overhead by ~10x compared to standard Python lists of strings.
    """
    def __init__(self, max_token_df=0.01, max_ngram_df=0.01):
        # 1% frequency threshold to filter out massive stopwords (e.g., 'limited', 'private')
        self.max_token_df = max_token_df
        self.max_ngram_df = max_ngram_df
        
        # Inverted Indexes (key -> array of target integer IDs)
        self.country_index = collections.defaultdict(lambda: array('i'))
        self.postal_index = collections.defaultdict(lambda: array('i'))
        self.token_index = collections.defaultdict(lambda: array('i'))
        self.ngram_index = collections.defaultdict(lambda: array('i'))
        
        # Document Frequencies (df)
        self.token_df = collections.defaultdict(int)
        self.ngram_df = collections.defaultdict(int)
        
        # Mapping from integer ID to actual string entity_id
        self.target_int_to_id = []
        self.total_targets = 0
        
        self.token_stopwords = set()
        self.ngram_stopwords = set()
        
        # IDF Dictionaries
        self.token_idf = {}
        self.ngram_idf = {}

    def add_target_batch(self, df):
        """
        Indexes a batch of target records (S2/S3).
        Expects a DataFrame with the normalized columns from Step 2.
        """
        start_int = self.total_targets
        
        # Extend the global ID map
        self.target_int_to_id.extend(df['entity_id'].tolist())
        
        # Use itertuples for rapid iteration over DataFrame
        for i, row in enumerate(df.itertuples(index=False)):
            int_id = start_int + i
            
            # Country
            country = getattr(row, 'country_normalized', None)
            if pd.notna(country) and country:
                self.country_index[country].append(int_id)
                
            # Postal
            postal = getattr(row, 'postal_code', None)
            if pd.notna(postal) and postal:
                self.postal_index[postal].append(int_id)
                
            # Tokens
            tokens_val = getattr(row, 'business_name_tokenized', [])
            if isinstance(tokens_val, list):
                # Unique tokens in this document
                for t in set(tokens_val):
                    self.token_index[t].append(int_id)
                    self.token_df[t] += 1
                
            # 4-grams
            compact = getattr(row, 'business_name_compact', "")
            if pd.notna(compact) and compact:
                for ng in set(get_4grams(str(compact))):
                    self.ngram_index[ng].append(int_id)
                    self.ngram_df[ng] += 1
                
        self.total_targets += len(df)
        
    def finalize_index(self):
        """
        Computes IDF values and prunes extremely common tokens (stopwords).
        Must be called after all batches are added.
        """
        print(f"Finalizing index for {self.total_targets} targets...")
        
        t_thresh = max(2, self.total_targets * self.max_token_df)
        n_thresh = max(2, self.total_targets * self.max_ngram_df)
        
        self.token_stopwords = {k for k, v in self.token_df.items() if v > t_thresh}
        self.ngram_stopwords = {k for k, v in self.ngram_df.items() if v > n_thresh}
        
        print(f"Pruned {len(self.token_stopwords)} highly frequent tokens.")
        print(f"Pruned {len(self.ngram_stopwords)} highly frequent ngrams.")
        
        for k, v in self.token_df.items():
            if k not in self.token_stopwords:
                self.token_idf[k] = math.log(self.total_targets / (v + 1))
                
        for k, v in self.ngram_df.items():
            if k not in self.ngram_stopwords:
                self.ngram_idf[k] = math.log(self.total_targets / (v + 1))
                
    def search(self, query_dict, top_k=30, active_paths=None):
        """
        Generates candidates for a single S1 record using active blocking paths.
        active_paths: set of strings, e.g., {'token', 'postal', 'country', 'ngram'}
        """
        if active_paths is None:
            active_paths = {'token', 'postal', 'country', 'ngram'}
            
        scores = collections.defaultdict(float)
        provenance = collections.defaultdict(set)
        
        # 1. Postal Block
        if 'postal' in active_paths:
            postal = query_dict.get('postal_code')
            if pd.notna(postal) and postal in self.postal_index:
                for tid in self.postal_index[postal]:
                    scores[tid] += 10.0  # High heuristic weight for exact PIN match
                    provenance[tid].add('postal')
                    
        # 2. Token Block (IDF Weighted)
        if 'token' in active_paths:
            tokens = query_dict.get('business_name_tokenized', [])
            if isinstance(tokens, list):
                for t in set(tokens):
                    if t in self.token_idf:
                        idf = self.token_idf[t]
                        for tid in self.token_index[t]:
                            scores[tid] += idf
                            provenance[tid].add('token')
                            
        # 3. N-gram Block (IDF Weighted)
        if 'ngram' in active_paths:
            compact = query_dict.get('business_name_compact', "")
            if pd.notna(compact):
                for ng in set(get_4grams(str(compact))):
                    if ng in self.ngram_idf:
                        idf = self.ngram_idf[ng]
                        for tid in self.ngram_index[ng]:
                            scores[tid] += idf * 0.3  # Down-weight ngrams relative to full tokens
                            provenance[tid].add('ngram')
                            
        # 4. Country Modulator (Bonus/Penalty instead of absolute dropping)
        if 'country' in active_paths:
            country = query_dict.get('country_normalized')
            if pd.notna(country) and country in self.country_index:
                valid_tids = set(self.country_index[country])
                # Modify existing scores
                for tid in list(scores.keys()):
                    if tid in valid_tids:
                        scores[tid] *= 1.5  # Bonus for matching country
                        provenance[tid].add('country')
                    else:
                        scores[tid] *= 0.2  # Heavy penalty, but not zeroing out (in case of missing/wrong country)

        if not scores:
            return []
            
        # Top-K Extraction
        # Sort manually; for K=30 this is fast enough if candidate pool < 100k
        top_tids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for tid, score in top_tids:
            results.append({
                'candidate_id': self.target_int_to_id[tid],
                'score': score,
                'provenance': ",".join(provenance[tid])
            })
            
        return results
