import os
import faiss
import numpy as np
import pandas as pd
import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

class DenseBlocker:
    """
    Production-ready FAISS HNSW Dense Blocker using Multilingual Sentence Transformers.
    """
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', device=None, index_path_prefix=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.device == 'cpu':
            torch.set_num_threads(os.cpu_count() or 8)
            
        print(f"Loading SentenceTransformer: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Mapping from FAISS internal integer ID (0 to N-1) to original string entity_id
        self.id_mapping = []
        
        # FAISS HNSW Index with Inner Product (for normalized vectors, this equals Cosine Similarity)
        # M=32 specifies the number of connections per layer in the HNSW graph
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        
        # Tuning HNSW parameters
        self.index.hnsw.efConstruction = 200 # Higher = slower build, better accuracy
        self.index.hnsw.efSearch = 128       # Higher = slower search, better recall
        
        if index_path_prefix and os.path.exists(f"{index_path_prefix}.index"):
            self.load_index(index_path_prefix)
            
    def format_record(self, row):
        """
        Creates a structured text representation for the model.
        Preserves Unicode naturally. 
        """
        name = getattr(row, 'business_name', '')
        addr = getattr(row, 'business_address', '')
        country = getattr(row, 'country', '')
        
        # Handle nan floats if present
        if pd.isna(name): name = ""
        if pd.isna(addr): addr = ""
        if pd.isna(country): country = ""
        
        return f"NAME: {name} | ADDRESS: {addr} | COUNTRY: {country}"

    def encode_batch(self, texts, batch_size=256):
        """
        Encodes a list of texts and L2 normalizes them for FAISS Inner Product (Cosine).
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        return embeddings

    def add_targets(self, df, batch_size=1024):
        """
        Encodes a dataframe of targets and adds them to the FAISS index.
        Memory conscious processing using batches.
        """
        texts = []
        ids = []
        
        # Collect formatted texts
        for row in df.itertuples():
            texts.append(self.format_record(row))
            ids.append(getattr(row, 'entity_id'))
            
        # Process in chunks
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Targets"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            embeddings = self.encode_batch(batch_texts, batch_size=batch_size)
            self.index.add(embeddings)
            self.id_mapping.extend(batch_ids)

    def search(self, df, k=30, batch_size=1024):
        """
        Searches the FAISS index for queries in df.
        Returns a dictionary: {query_entity_id: [candidate_entity_ids]}
        Also returns scores for analysis.
        """
        texts = []
        query_ids = []
        
        for row in df.itertuples():
            texts.append(self.format_record(row))
            query_ids.append(getattr(row, 'entity_id'))
            
        results = {}
        for i in tqdm(range(0, len(texts), batch_size), desc="Searching FAISS"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = query_ids[i:i+batch_size]
            
            embeddings = self.encode_batch(batch_texts, batch_size=batch_size)
            distances, indices = self.index.search(embeddings, k)
            
            for j in range(len(batch_ids)):
                q_id = batch_ids[j]
                # Filter out -1 (FAISS returns -1 if there aren't enough elements in the index)
                cands = []
                for idx in indices[j]:
                    if idx != -1:
                        cands.append(self.id_mapping[idx])
                results[q_id] = cands
                
        return results

    def save_index(self, path_prefix):
        """
        Saves FAISS index and the critical ID mapping list.
        """
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}_mapping.json", 'w') as f:
            json.dump(self.id_mapping, f)
        print(f"FAISS index and metadata saved to {path_prefix}.index")
            
    def load_index(self, path_prefix):
        """
        Loads FAISS index and ID mapping.
        """
        print(f"Loading FAISS index from {path_prefix}.index")
        self.index = faiss.read_index(f"{path_prefix}.index")
        with open(f"{path_prefix}_mapping.json", 'r') as f:
            self.id_mapping = json.load(f)
        print(f"Loaded {len(self.id_mapping)} target entities.")
