import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

from src.business_entity_resolution.preprocessing import preprocess_dataframe
from src.business_entity_resolution.sparse_blocking import SparseBlocker
from src.business_entity_resolution.dense_blocking import DenseBlocker
from src.business_entity_resolution.candidate_generation import CandidateUnion
from src.business_entity_resolution.features import generate_all_features
from src.business_entity_resolution.postprocessing import build_singleton_features, apply_one_to_one_constraint

class InferencePipeline:
    """
    Production-ready end-to-end inference pipeline for Business Entity Resolution.
    Performs preprocessing, sparse/dense candidate blocking, candidate union,
    feature engineering, XGBoost prediction, singleton gating, 1-to-1 constraint
    enforcement, and final submission formatting.
    """
    def __init__(self, test_dir='dataset/test', output_dir='output', models_dir='models', exp_dir='experiments'):
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.exp_dir = exp_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Verify required artifacts
        self.verify_artifacts()
        
        # Load postprocessing configuration
        with open(os.path.join(self.exp_dir, 'postprocessing_metrics.json'), 'r') as f:
            self.metrics_cfg = json.load(f)
            
        self.sg_thresh = self.metrics_cfg.get('singleton_gate_threshold', 0.5)
        self.match_thresh = self.metrics_cfg.get('final_match_threshold', 0.86)
        self.one_to_one_enabled = self.metrics_cfg.get('one_to_one_constraint_verified', True)
        
        # Load XGBoost model
        xgb_model_path = os.path.join(self.models_dir, 'xgboost', 'entity_matcher_hard_negative.json')
        if not os.path.exists(xgb_model_path):
            xgb_model_path = os.path.join(self.models_dir, 'xgboost', 'entity_matcher.json')
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(xgb_model_path)
        
        # Load feature schema
        with open(os.path.join(self.exp_dir, 'feature_schema.json'), 'r') as f:
            self.feature_names = json.load(f)['features']
            
        # Load Singleton Gate model
        sg_model_path = os.path.join(self.models_dir, 'singleton', 'singleton_gate_model.joblib')
        self.sg_model = joblib.load(sg_model_path)
        
    def verify_artifacts(self):
        required = [
            os.path.join(self.models_dir, 'xgboost', 'entity_matcher_hard_negative.json'),
            os.path.join(self.exp_dir, 'feature_schema.json'),
            os.path.join(self.models_dir, 'singleton', 'singleton_gate_model.joblib'),
            os.path.join(self.exp_dir, 'postprocessing_metrics.json')
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing required pipeline artifacts: {missing}")
        print("All required inference artifacts verified.")

    def run_full_pipeline(self, max_s1_test=None, max_target_test=None):
        t0 = time.time()
        print("=== STEP 10: RUNNING FINAL TEST INFERENCE PIPELINE ===")
        
        # 1. Load Test Datasets
        print("Loading test datasets...")
        s1_path = os.path.join(self.test_dir, 'test_source1.tsv')
        s2_path = os.path.join(self.test_dir, 'test_source2.tsv')
        s3_path = os.path.join(self.test_dir, 'test_source3.tsv')
        
        df_s1 = pd.read_csv(s1_path, sep='\t', dtype=str, nrows=max_s1_test)
        df_s2 = pd.read_csv(s2_path, sep='\t', dtype=str, nrows=max_target_test)
        df_s3 = pd.read_csv(s3_path, sep='\t', dtype=str, nrows=max_target_test)
        
        s1_count, s2_count, s3_count = len(df_s1), len(df_s2), len(df_s3)
        print(f"Loaded S1: {s1_count}, S2: {s2_count}, S3: {s3_count}")
        
        # 2. Preprocess & Normalize
        print("Preprocessing test data...")
        df_s1_norm = preprocess_dataframe(df_s1)
        df_s2_norm = preprocess_dataframe(df_s2)
        df_s3_norm = preprocess_dataframe(df_s3)
        
        df_targets = pd.concat([df_s2_norm, df_s3_norm], ignore_index=True)
        
        # 3. Sparse Blocking
        print("Running Sparse Blocking...")
        blocker = SparseBlocker(max_token_df=0.01, max_ngram_df=0.01)
        blocker.add_target_batch(df_targets)
        blocker.finalize_index()
        
        sparse_cand_dict = {}
        for row in df_s1_norm.itertuples(index=False):
            q_dict = {
                'entity_id': getattr(row, 'entity_id'),
                'business_name_tokenized': getattr(row, 'business_name_tokenized', []),
                'business_name_compact': getattr(row, 'business_name_compact', ""),
                'postal_code': getattr(row, 'postal_code', None),
                'country_normalized': getattr(row, 'country_normalized', None)
            }
            cands = blocker.search(q_dict, top_k=30)
            sparse_cand_dict[q_dict['entity_id']] = set([c['candidate_id'] for c in cands])
            
        # 4. Dense FAISS Blocking
        print("Running Dense FAISS Blocking...")
        dense_index_prefix = os.path.join(self.models_dir, 'faiss', 'target_index_v1')
        dense_blocker = DenseBlocker(index_path_prefix=dense_index_prefix if os.path.exists(f"{dense_index_prefix}.index") else None)
        
        if not os.path.exists(f"{dense_index_prefix}.index"):
            dense_blocker.add_targets(df_targets, batch_size=2048)
            
        dense_cand_dict = dense_blocker.search(df_s1_norm, k=30, batch_size=2048)
        dense_cand_dict_set = {k: set(v) for k, v in dense_cand_dict.items()}
        
        # 5. Candidate Union & Save candidate_pairs.tsv
        print("Unifying Candidates...")
        union_dict = CandidateUnion.generate_union_provenance(sparse_cand_dict, dense_cand_dict_set)
        pruned_dict = CandidateUnion.prune_candidates(union_dict, max_k=40)
        
        cand_pairs_path = os.path.join(self.output_dir, 'candidate_pairs.tsv')
        CandidateUnion.save_final_candidates(pruned_dict, cand_pairs_path)
        print(f"Saved candidate_pairs.tsv to {cand_pairs_path}")
        
        # Prepare candidate pairs DataFrame for Feature Engineering
        pair_rows = []
        for s1_id in sorted(pruned_dict.keys()):
            for cand_id, prov in pruned_dict[s1_id]:
                pair_rows.append({
                    'source1_entity_id': s1_id,
                    'candidate_entity_id': cand_id,
                    'provenance': prov
                })
        df_pairs = pd.DataFrame(pair_rows)
        if df_pairs.empty:
            df_pairs = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'provenance'])
            
        # 6. Feature Engineering
        print(f"Generating pairwise features for {len(df_pairs)} candidate pairs...")
        if not df_pairs.empty:
            # Merge S1 fields
            df_merged = df_pairs.merge(
                df_s1_norm[['entity_id', 'business_name', 'business_address', 'country', 'postal_code']].rename(columns={
                    'entity_id': 's1_id',
                    'business_name': 's1_name',
                    'business_address': 's1_address',
                    'country': 's1_country',
                    'postal_code': 's1_postal_code'
                }),
                left_on='source1_entity_id',
                right_on='s1_id',
                how='left'
            ).merge(
                df_targets[['entity_id', 'business_name', 'business_address', 'country', 'postal_code']].rename(columns={
                    'entity_id': 'cand_id',
                    'business_name': 'cand_name',
                    'business_address': 'cand_address',
                    'country': 'cand_country',
                    'postal_code': 'cand_postal_code'
                }),
                left_on='candidate_entity_id',
                right_on='cand_id',
                how='left'
            )
            df_features = generate_all_features(df_merged)
        else:
            df_features = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'] + self.feature_names)
            
        # 7. XGBoost Inference
        print("Predicting matching probabilities with XGBoost...")
        if not df_features.empty:
            dmatrix = xgb.DMatrix(df_features[self.feature_names])
            df_features['probability'] = self.xgb_model.predict(dmatrix)
            df_features['label'] = 0  # placeholder for postprocessing features
        else:
            df_features['probability'] = []
            df_features['label'] = []
            
        # 8. Singleton Gate
        print("Applying Singleton Gate...")
        if not df_features.empty:
            single_features = build_singleton_features(df_features)
            feature_cols = [c for c in single_features.columns if c not in ['source1_entity_id', 'is_singleton']]
            single_features['singleton_prob'] = self.sg_model.predict_proba(single_features[feature_cols])[:, 1]
            
            gated_entities = set(single_features[single_features['singleton_prob'] >= self.sg_thresh]['source1_entity_id'])
        else:
            single_features = pd.DataFrame()
            gated_entities = set()
            
        # 9. One-to-One Constraint & Match Thresholding
        print("Applying 1-to-1 Constraint & Final Thresholding...")
        if not df_features.empty and self.one_to_one_enabled:
            df_processed = apply_one_to_one_constraint(df_features)
            df_processed['singleton_decision'] = df_processed['source1_entity_id'].isin(gated_entities)
            df_processed['final_match'] = (
                (df_processed['probability'] >= self.match_thresh) &
                (~df_processed['singleton_decision']) &
                (df_processed['uniqueness_decision'] == True)
            )
        elif not df_features.empty:
            df_processed = df_features.copy()
            df_processed['singleton_decision'] = df_processed['source1_entity_id'].isin(gated_entities)
            df_processed['final_match'] = (
                (df_processed['probability'] >= self.match_thresh) &
                (~df_processed['singleton_decision'])
            )
        else:
            df_processed = df_features.copy()
            df_processed['final_match'] = []
            
        # 10. Format and Write matching_results.tsv
        print("Writing matching_results.tsv...")
        all_s1_ids = df_s1['entity_id'].tolist()
        
        matches_dict = {s1_id: [] for s1_id in all_s1_ids}
        if not df_processed.empty:
            matched_pairs = df_processed[df_processed['final_match'] == True]
            for s1_id, cands in matched_pairs.groupby('source1_entity_id')['candidate_entity_id'].apply(list).items():
                matches_dict[s1_id] = cands
                
        out_matching_rows = []
        for s1_id in sorted(all_s1_ids):
            cands = matches_dict.get(s1_id, [])
            out_matching_rows.append({
                'source1_entity_id': s1_id,
                'matched_entity_ids': ",".join(cands)
            })
            
        matching_results_path = os.path.join(self.output_dir, 'matching_results.tsv')
        df_matching_out = pd.DataFrame(out_matching_rows)
        df_matching_out.to_csv(matching_results_path, sep='\t', index=False)
        print(f"Saved matching_results.tsv to {matching_results_path}")
        
        # 11. Final Sanity Summary Metrics
        total_candidates = len(df_pairs)
        candidate_counts = [len(pruned_dict.get(s1_id, [])) for s1_id in all_s1_ids]
        
        total_matches = sum(len(c) for c in matches_dict.values())
        s2_matches = sum(sum(1 for cid in c if cid.startswith('S2')) for c in matches_dict.values())
        s3_matches = sum(sum(1 for cid in c if cid.startswith('S3')) for c in matches_dict.values())
        
        summary = {
            's1_test_count': s1_count,
            's2_test_count': s2_count,
            's3_test_count': s3_count,
            'total_candidate_pairs': total_candidates,
            'average_candidates_per_s1': float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            'median_candidates_per_s1': float(np.median(candidate_counts)) if candidate_counts else 0.0,
            'p95_candidates_per_s1': float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0,
            'p99_candidates_per_s1': float(np.percentile(candidate_counts, 99)) if candidate_counts else 0.0,
            'max_candidates_per_s1': int(np.max(candidate_counts)) if candidate_counts else 0,
            'zero_candidate_s1_count': int(sum(1 for c in candidate_counts if c == 0)),
            'predicted_singleton_count': len(gated_entities),
            'total_predicted_matches': total_matches,
            's2_matches_count': s2_matches,
            's3_matches_count': s3_matches,
            'singleton_threshold': self.sg_thresh,
            'final_match_threshold': self.match_thresh,
            'one_to_one_constraint_enabled': self.one_to_one_enabled,
            'pipeline_execution_time_seconds': float(time.time() - t0)
        }
        
        summary_path = os.path.join(self.exp_dir, 'final_test_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary written to {summary_path}")
        print("=== STEP 10 INFERENCE COMPLETED SUCCESSFULLY ===")
        return summary
