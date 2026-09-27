import pandas as pd
import collections

class CandidateUnion:
    """
    Utility for unifying, pruning, and evaluating candidates from 
    independent blocking paths (Sparse and Dense) into a final Tier 1 Candidate Set.
    """
    
    @staticmethod
    def load_candidate_dict(filepath):
        """
        Loads a candidate TSV file into a dictionary format: {source1_id: set(candidate_ids)}
        """
        df = pd.read_csv(filepath, sep='\t', dtype=str)
        cand_dict = {}
        for _, row in df.iterrows():
            s1 = row['source1_entity_id']
            cands = row['candidate_entity_ids']
            if pd.isna(cands) or not cands:
                cand_dict[s1] = set()
            else:
                cand_dict[s1] = set(cands.split(','))
        return cand_dict

    @staticmethod
    def generate_union_provenance(sparse_cands, dense_cands):
        """
        Takes sparse and dense candidate dictionaries.
        Returns a structured dictionary mapping S1 -> list of (TargetID, Provenance).
        Provenance is 'SPARSE', 'DENSE', or 'BOTH'.
        """
        all_s1_keys = set(sparse_cands.keys()).union(set(dense_cands.keys()))
        
        union_dict = {}
        
        for s1 in all_s1_keys:
            s_set = sparse_cands.get(s1, set())
            d_set = dense_cands.get(s1, set())
            
            final_cands = []
            
            # Intersection (BOTH)
            both = s_set.intersection(d_set)
            for c in both:
                final_cands.append((c, 'BOTH'))
                
            # Sparse Only
            for c in s_set.difference(d_set):
                final_cands.append((c, 'SPARSE'))
                
            # Dense Only
            for c in d_set.difference(s_set):
                final_cands.append((c, 'DENSE'))
                
            union_dict[s1] = final_cands
            
        return union_dict

    @staticmethod
    def prune_candidates(union_dict, max_k=40):
        """
        Prunes candidates if an S1 entity exceeds max_k.
        Prioritizes 'BOTH' > 'SPARSE' > 'DENSE' logically if we have to cut.
        In practice, 'SPARSE' (exact matches) and 'BOTH' are extremely high confidence.
        """
        pruned_dict = {}
        for s1, cands in union_dict.items():
            if len(cands) <= max_k:
                pruned_dict[s1] = cands
                continue
                
            # If we need to prune, sort by provenance priority
            # BOTH (0), SPARSE (1), DENSE (2)
            priority = {'BOTH': 0, 'SPARSE': 1, 'DENSE': 2}
            sorted_cands = sorted(cands, key=lambda x: priority[x[1]])
            
            pruned_dict[s1] = sorted_cands[:max_k]
            
        return pruned_dict

    @staticmethod
    def save_final_candidates(union_dict, filepath):
        """
        Saves the final candidate set to TSV format exactly as expected for Step 5.
        """
        out_rows = []
        # Sort S1 keys for deterministic output
        for s1 in sorted(union_dict.keys()):
            cands = union_dict[s1]
            # Strip provenance for the final output, keep deterministic ordering
            cand_ids = [c[0] for c in cands]
            # Ensure uniqueness
            unique_cand_ids = list(dict.fromkeys(cand_ids))
            
            out_rows.append({
                'source1_entity_id': s1,
                'candidate_entity_ids': ",".join(unique_cand_ids)
            })
            
        pd.DataFrame(out_rows).to_csv(filepath, sep='\t', index=False)
