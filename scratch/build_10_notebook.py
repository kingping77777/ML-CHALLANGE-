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

# Title
cells.append(create_markdown_cell("# STEP 10 — FINAL TEST INFERENCE + OUTPUT GENERATION + SUBMISSION PACKAGE\n\nThis notebook executes the full, validated end-to-end entity resolution pipeline on the unseen test dataset."))

# 1. Setup & Imports
cells.append(create_markdown_cell("### 1. Imports & Environment Setup"))
code_1 = """import os
import sys

if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')

for p in ['.', '..', '../..', 'src']:
    abs_p = os.path.abspath(p)
    if abs_p not in sys.path:
        sys.path.append(abs_p)

import json
import pandas as pd
import numpy as np

from src.business_entity_resolution.pipeline import InferencePipeline

test_dir = 'dataset/test'
output_dir = 'output'
exp_dir = 'experiments'
models_dir = 'models'
"""
cells.append(create_code_cell(code_1))

# 2. Verify Artifacts
cells.append(create_markdown_cell("### 2. Verify Inference Artifacts"))
code_2 = """pipeline = InferencePipeline(
    test_dir=test_dir,
    output_dir=output_dir,
    models_dir=models_dir,
    exp_dir=exp_dir
)

print(f"Selected Singleton Gate Threshold: {pipeline.sg_thresh}")
print(f"Selected Match Threshold: {pipeline.match_thresh}")
print(f"1-to-1 Constraint Enabled: {pipeline.one_to_one_enabled}")
"""
cells.append(create_code_cell(code_2))

# 3. Test Data Validation
cells.append(create_markdown_cell("### 3. Load & Validate Test Datasets"))
code_3 = """s1_df = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\\t', dtype=str)
s2_df = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\\t', dtype=str)
s3_df = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\\t', dtype=str)

print(f"Test S1 entities: {len(s1_df)} (Unique IDs: {s1_df['entity_id'].nunique()})")
print(f"Test S2 entities: {len(s2_df)} (Unique IDs: {s2_df['entity_id'].nunique()})")
print(f"Test S3 entities: {len(s3_df)} (Unique IDs: {s3_df['entity_id'].nunique()})")

print("Countries represented in test S1:", s1_df['country'].value_counts().to_dict())
"""
cells.append(create_code_cell(code_3))

# 4. Run Inference Pipeline
cells.append(create_markdown_cell("### 4. Execute End-to-End Test Inference Pipeline"))
code_4 = """summary = pipeline.run_full_pipeline()
print("\\nSummary Metrics:")
print(json.dumps(summary, indent=2))
"""
cells.append(create_code_cell(code_4))

# 5. Programmatic Output Integrity Checks
cells.append(create_markdown_cell("### 5. Programmatic Output Integrity Checks"))
code_5 = """matching_path = os.path.join(output_dir, 'matching_results.tsv')
candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')

matching_df = pd.read_csv(matching_path, sep='\\t', dtype=str, keep_default_na=False)
candidate_df = pd.read_csv(candidate_path, sep='\\t', dtype=str, keep_default_na=False)

assert len(matching_df) == len(s1_df), f"Matching row count mismatch! Expected {len(s1_df)}, got {len(matching_df)}"
assert len(candidate_df) == len(s1_df), f"Candidate row count mismatch! Expected {len(s1_df)}, got {len(candidate_df)}"

assert matching_df['source1_entity_id'].nunique() == len(s1_df), "Duplicate S1 IDs in matching_results.tsv!"
assert candidate_df['source1_entity_id'].nunique() == len(s1_df), "Duplicate S1 IDs in candidate_pairs.tsv!"

print("Programmatic checks passed successfully!")
"""
cells.append(create_code_cell(code_5))

# 6. Run Official Submission Validator
cells.append(create_markdown_cell("### 6. Run Official Challenge Validator"))
code_6 = """import subprocess

validator_script = 'student_resource/utils/validate_submission.py'
cmd = [
    sys.executable, validator_script,
    '--matching', matching_path,
    '--candidate', candidate_path,
    '--test-dir', test_dir,
    '--check-ids'
]

print("Running command:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print("Validator STDOUT:\\n", res.stdout)
if res.stderr:
    print("Validator STDERR:\\n", res.stderr)

assert res.returncode == 0, "Official validator failed!"
print("OFFICIAL VALIDATOR PASSED SUCCESSFULLY!")
"""
cells.append(create_code_cell(code_6))

nb = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs('notebooks', exist_ok=True)
with open("notebooks/10_Final_Test_Inference.ipynb", "w", encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook 10 created successfully.")
