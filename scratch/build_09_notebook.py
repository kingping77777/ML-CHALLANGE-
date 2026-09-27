import json
import os

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split('\n')]
    }

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }

cells = []

cells.append(create_markdown_cell("# Step 9: Singleton Gate + 1-to-1 Constraint + Threshold Optimization"))

code = """import os
import sys

if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')

import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

for p in ['.', '..', '../..', 'src']:
    abs_p = os.path.abspath(p)
    if abs_p not in sys.path:
        sys.path.append(abs_p)

from src.business_entity_resolution.model import calculate_entity_macro_f05_from_dicts
from src.business_entity_resolution.postprocessing import build_singleton_features, apply_one_to_one_constraint

# Paths
models_dir = 'models/xgboost'
singleton_dir = 'models/singleton'
exp_dir = 'experiments'
raw_dir = 'dataset/train'

os.makedirs(singleton_dir, exist_ok=True)"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 2. Load Predictions & Model"))
code = """# Load Validation Data
val_df = pd.read_parquet(os.path.join(exp_dir, 'validation_pair_features.parquet'))
train_df = pd.read_parquet(os.path.join(exp_dir, 'train_pair_features.parquet'))

with open(os.path.join(exp_dir, 'feature_schema.json'), 'r') as f:
    feature_schema = json.load(f)
features = feature_schema['features']

# Load Model
hn_model = xgb.Booster()
hn_model.load_model(os.path.join(models_dir, 'entity_matcher_hard_negative.json'))

dtrain = xgb.DMatrix(train_df[features])
dval = xgb.DMatrix(val_df[features])

train_df['probability'] = hn_model.predict(dtrain)
val_df['probability'] = hn_model.predict(dval)

print(f"Scored {len(train_df)} training pairs and {len(val_df)} validation pairs.")"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 4. Build Entity-level Singleton Features & Train Gate"))
code = """print("Building features for training...")
train_single_features = build_singleton_features(train_df)

print("Building features for validation...")
val_single_features = build_singleton_features(val_df)

singleton_feat_cols = ['max_prob', 'mean_prob', 'median_prob', 'std_prob', 'candidate_count', 'high_prob_count', 'top1_minus_top2', 'top1_minus_mean']

X_train_sg = train_single_features[singleton_feat_cols]
y_train_sg = train_single_features['is_singleton']

X_val_sg = val_single_features[singleton_feat_cols]
y_val_sg = val_single_features['is_singleton']

# Train Lightweight Classifier
sg_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight='balanced')
sg_model.fit(X_train_sg, y_train_sg)

train_single_features['singleton_prob'] = sg_model.predict_proba(X_train_sg)[:, 1]
val_single_features['singleton_prob'] = sg_model.predict_proba(X_val_sg)[:, 1]

print("Singleton Gate trained.")"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 6. Evaluate Singleton Gate Thresholds"))
code = """# Pre-compute base matches once
pred_matches_base = val_df[val_df['probability'] >= 0.5].groupby('source1_entity_id')['candidate_entity_id'].apply(list).to_dict()

def evaluate_sg(val_df, val_single_features, sg_thresh):
    
    # Gate entities
    gated_entities = set(val_single_features[val_single_features['singleton_prob'] >= sg_thresh]['source1_entity_id'])
    
    # Apply gate
    pred_matches_gated = {k: v for k, v in pred_matches_base.items() if k not in gated_entities}
    
    y_true_dict = val_df[val_df['label'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(list).to_dict()
    all_s1 = val_df['source1_entity_id'].unique()
    y_true_dict = {s1: y_true_dict.get(s1, []) for s1 in all_s1}
    
    base_macro = calculate_entity_macro_f05_from_dicts(y_true_dict, {s1: pred_matches_base.get(s1, []) for s1 in all_s1})
    gated_macro = calculate_entity_macro_f05_from_dicts(y_true_dict, {s1: pred_matches_gated.get(s1, []) for s1 in all_s1})
    
    # Stats
    true_singletons = set(val_single_features[val_single_features['is_singleton'] == 1]['source1_entity_id'])
    true_non_singletons = set(all_s1) - true_singletons
    
    correct_gates = len(gated_entities.intersection(true_singletons))
    incorrect_gates = len(gated_entities.intersection(true_non_singletons))
    
    return {
        'threshold': sg_thresh,
        'predicted_singletons': len(gated_entities),
        'correctly_gated': correct_gates,
        'incorrectly_gated': incorrect_gates,
        'base_macro_f05': base_macro,
        'gated_macro_f05': gated_macro
    }

results = []
for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    results.append(evaluate_sg(val_df, val_single_features, t))

sg_eval_df = pd.DataFrame(results)
print(sg_eval_df)"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 7. Verify S2/S3 Uniqueness Constraint"))
code = """train_raw = pd.read_csv(os.path.join(raw_dir, 'train_ground_truth.tsv'), sep='	')
s2_s3_matches = train_raw['matched_entity_ids'].dropna().astype(str).str.split('|').explode().str.strip().tolist()

num_total = len(s2_s3_matches)
num_unique = len(set(s2_s3_matches))

print(f"Total matched candidates: {num_total}")
print(f"Unique matched candidates: {num_unique}")

if num_total == num_unique:
    print("VERIFIED: One-to-one constraint holds strictly on the training set.")
    one_to_one_verified = True
else:
    print("WARNING: One-to-one constraint violated. Duplicates exist.")
    one_to_one_verified = False"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 8. Measure Uniqueness Effect"))
code = """if one_to_one_verified:
    # 1. XGBoost only (Threshold = 0.5)
    pred_base = val_df[val_df['probability'] >= 0.5]
    y_true_dict = val_df[val_df['label'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(list).to_dict()
    all_s1 = val_df['source1_entity_id'].unique()
    y_true_dict = {s1: y_true_dict.get(s1, []) for s1 in all_s1}
    
    base_matches = {s1: pred_base[pred_base['source1_entity_id'] == s1]['candidate_entity_id'].tolist() for s1 in all_s1}
    base_macro = calculate_entity_macro_f05_from_dicts(y_true_dict, base_matches)
    
    # 2. XGBoost + 1-to-1
    val_1_to_1 = apply_one_to_one_constraint(val_df)
    pred_1_to_1 = val_1_to_1[(val_1_to_1['probability'] >= 0.5) & (val_1_to_1['uniqueness_decision'] == True)]
    
    oto_matches = {s1: pred_1_to_1[pred_1_to_1['source1_entity_id'] == s1]['candidate_entity_id'].tolist() for s1 in all_s1}
    oto_macro = calculate_entity_macro_f05_from_dicts(y_true_dict, oto_matches)
    
    duplicates_before = len(pred_base) - len(pred_base['candidate_entity_id'].unique())
    removed = len(pred_base) - len(pred_1_to_1)
    
    # Calculate True/False removed
    removed_df = val_1_to_1[(val_1_to_1['probability'] >= 0.5) & (val_1_to_1['uniqueness_decision'] == False)]
    true_removed = len(removed_df[removed_df['label'] == 1])
    false_removed = len(removed_df[removed_df['label'] == 0])
    
    print(f"Duplicates before: {duplicates_before}")
    print(f"Assignments removed: {removed} (True: {true_removed}, False: {false_removed})")
    print(f"Base Macro F0.5: {base_macro:.4f}")
    print(f"1-to-1 Macro F0.5: {oto_macro:.4f}")"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 10. F0.5 Threshold Sweep"))
code = """# First, select best SG threshold based on macro F0.5 gain
best_sg_thresh = sg_eval_df.sort_values(by='gated_macro_f05', ascending=False).iloc[0]['threshold']
print(f"Selected Singleton Gate Threshold: {best_sg_thresh}")

gated_entities = set(val_single_features[val_single_features['singleton_prob'] >= best_sg_thresh]['source1_entity_id'])

# Apply Gate and Uniqueness logic beforehand
val_processed = apply_one_to_one_constraint(val_df)
val_processed['singleton_decision'] = val_processed['source1_entity_id'].isin(gated_entities)
val_processed['valid_candidate'] = (~val_processed['singleton_decision']) & (val_processed['uniqueness_decision'] == True)

y_true_dict = val_df[val_df['label'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(list).to_dict()
all_s1 = val_df['source1_entity_id'].unique()
y_true_dict = {s1: y_true_dict.get(s1, []) for s1 in all_s1}

sweep_results = []

for match_thresh in np.arange(0.10, 1.00, 0.01):
    pred_df = val_processed[(val_processed['probability'] >= match_thresh) & (val_processed['valid_candidate'] == True)]
    
    matches_dict = {s1: [] for s1 in all_s1}
    for s1, cands in pred_df.groupby('source1_entity_id')['candidate_entity_id'].apply(list).items():
        matches_dict[s1] = cands
        
    macro_f05 = calculate_entity_macro_f05_from_dicts(y_true_dict, matches_dict)
    
    # Calculate Pairwise
    tp = len(pred_df[pred_df['label'] == 1])
    fp = len(pred_df[pred_df['label'] == 0])
    fn = len(val_df[(val_df['label'] == 1) & (~val_df.index.isin(pred_df.index))])
    
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    pair_f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if prec + rec > 0 else 0
    
    predicted_matches = len(pred_df)
    predicted_singletons = len([s1 for s1, cands in matches_dict.items() if len(cands) == 0])
    
    true_singletons = set(val_single_features[val_single_features['is_singleton'] == 1]['source1_entity_id'])
    fp_singletons = len(set(s1 for s1, cands in matches_dict.items() if len(cands) == 0) - true_singletons)
    sg_fpr = fp_singletons / len(set(all_s1) - true_singletons) if len(set(all_s1) - true_singletons) > 0 else 0
    
    sweep_results.append({
        'threshold': match_thresh,
        'predicted_match_count': predicted_matches,
        'predicted_singleton_count': predicted_singletons,
        'pairwise_precision': prec,
        'pairwise_recall': rec,
        'pairwise_f0.5': pair_f05,
        'macro_f0.5': macro_f05,
        'singleton_false_positive_rate': sg_fpr,
        'candidate_conflict_count': len(val_processed[(val_processed['probability'] >= match_thresh)]) - len(val_processed[(val_processed['probability'] >= match_thresh) & (val_processed['uniqueness_decision'] == True)]),
        'removed_by_uniqueness': len(val_processed[(val_processed['probability'] >= match_thresh) & (val_processed['uniqueness_decision'] == False)])
    })

sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(os.path.join(exp_dir, 'threshold_sweep.csv'), index=False)

best_row = sweep_df.sort_values('macro_f0.5', ascending=False).iloc[0]
best_match_thresh = best_row['threshold']
print(f"Best Match Threshold: {best_match_thresh:.2f} (Macro F0.5: {best_row['macro_f0.5']:.4f})")"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 11. Compare All Variants"))
code = """def evaluate_variant(name, pred_df):
    matches_dict = {s1: [] for s1 in all_s1}
    for s1, cands in pred_df.groupby('source1_entity_id')['candidate_entity_id'].apply(list).items():
        matches_dict[s1] = cands
    
    macro_f05 = calculate_entity_macro_f05_from_dicts(y_true_dict, matches_dict)
    
    tp = len(pred_df[pred_df['label'] == 1])
    fp = len(pred_df[pred_df['label'] == 0])
    fn = len(val_df[(val_df['label'] == 1) & (~val_df.index.isin(pred_df.index))])
    
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    
    true_singletons = set(val_single_features[val_single_features['is_singleton'] == 1]['source1_entity_id'])
    fp_singletons = len(set(s1 for s1, cands in matches_dict.items() if len(cands) == 0) - true_singletons)
    sg_fpr = fp_singletons / len(set(all_s1) - true_singletons) if len(set(all_s1) - true_singletons) > 0 else 0
    
    return {
        'variant': name,
        'macro_F0.5': macro_f05,
        'precision': prec,
        'recall': rec,
        'singleton_false_positive_rate': sg_fpr,
        'predicted_matches': len(pred_df)
    }

variants = []

# A: XGBoost Only (at 0.5)
variants.append(evaluate_variant('A: XGBoost Only', val_df[val_df['probability'] >= 0.5]))

# B: XGBoost + Gate
variants.append(evaluate_variant('B: XGBoost + SG', val_processed[(val_processed['probability'] >= 0.5) & (~val_processed['singleton_decision'])]))

# C: XGBoost + 1-to-1
variants.append(evaluate_variant('C: XGBoost + 1-to-1', val_processed[(val_processed['probability'] >= 0.5) & (val_processed['uniqueness_decision'])]))

# D: XGBoost + Gate + 1-to-1
variants.append(evaluate_variant('D: XGBoost + SG + 1-to-1', val_processed[(val_processed['probability'] >= 0.5) & (val_processed['valid_candidate'])]))

# E: Optimized
variants.append(evaluate_variant('E: XGBoost + SG + 1-to-1 + Optimized Thresh', val_processed[(val_processed['probability'] >= best_match_thresh) & (val_processed['valid_candidate'])]))

comp_df = pd.DataFrame(variants)
comp_df.to_csv(os.path.join(exp_dir, 'postprocessing_comparison.csv'), index=False)
print(comp_df)"""
cells.append(create_code_cell(code))

cells.append(create_markdown_cell("### 12. Save Artifacts"))
code = """import joblib
joblib.dump(sg_model, os.path.join(singleton_dir, 'singleton_gate_model.joblib'))

val_processed['final_match_decision'] = (val_processed['probability'] >= best_match_thresh) & (val_processed['valid_candidate'])

val_processed.to_parquet(os.path.join(exp_dir, 'final_validation_predictions.parquet'), index=False)

metrics = {
    'singleton_gate_threshold': float(best_sg_thresh),
    'final_match_threshold': float(best_match_thresh),
    'validation_macro_f0.5': float(best_row['macro_f0.5']),
    'xgboost_model_version': 'entity_matcher_hard_negative',
    'gate_configuration': 'RandomForest_50_5',
    'one_to_one_constraint_verified': bool(one_to_one_verified)
}
with open(os.path.join(exp_dir, 'postprocessing_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)
    
print("Artifacts saved successfully!")"""
cells.append(create_code_cell(code))

nb = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("c:/Users/Daksh/Downloads/6ab10eb3b23ba_student_resource/notebooks/09_Postprocessing_and_Threshold.ipynb", "w", encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook 09 created successfully.")
