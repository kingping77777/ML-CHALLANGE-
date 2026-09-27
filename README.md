# Amazon ML Challenge 2026 — Business Entity Resolution Pipeline

This repository contains the end-to-end, reproducible solution for the Amazon ML Challenge 2026 Business Entity Resolution task.

## 1. Environment Setup

- **Python Version**: Python 3.8+ (Tested on Python 3.11/3.13)
- **Virtual Environment Setup**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

## 2. Input Data Locations

Place input dataset files in the `dataset/` directory:
- `dataset/test/test_source1.tsv`
- `dataset/test/test_source2.tsv`
- `dataset/test/test_source3.tsv`

## 3. Pipeline Architecture

1. **Text Normalization**: Standardizes company names, addresses, and postal codes while preserving multilingual scripts (Devanagari, Cyrillic, etc.) and original attributes.
2. **Sparse Candidate Blocking**: Uses an inverted index over 4-grams, tokens, and country-constrained postal codes.
3. **Dense Candidate Blocking**: Embeds records using `paraphrase-multilingual-MiniLM-L12-v2` and retrieves K-nearest candidates using a FAISS HNSW graph index.
4. **Candidate Union**: Combines sparse and dense candidate sets into a high-recall candidate pool (`output/candidate_pairs.tsv`).
5. **Feature Engineering**: Computes string similarity metrics (Levenshtein, Jaro-Winkler, Token Sort, Token/N-gram Jaccard, Overlap), structured address similarities, dense embedding similarities, metadata indicators, and missingness flags.
6. **XGBoost Pairwise Model**: Predicts matching probability $P(\text{SAME BUSINESS})$ using the hard-negative-mined XGBoost classifier (`models/xgboost/entity_matcher_hard_negative.json`).
7. **Singleton Gate**: Filters out entities with no valid matches using a trained entity-level RandomForest classifier (`models/singleton/singleton_gate_model.joblib`) at threshold $0.5$.
8. **One-to-One Constraint Enforcement**: Ensures each S2/S3 candidate is assigned to at most one S1 query by preserving the highest probability assignment.
9. **F0.5 Match Thresholding**: Applies final probability threshold $0.86$ optimized on validation macro $F_{0.5}$.
10. **Output Generation**: Writes final predictions to `output/matching_results.tsv`.

## 4. How to Run Final Test Inference

To execute end-to-end inference and produce submission files:

```python
from src.business_entity_resolution.pipeline import InferencePipeline

pipeline = InferencePipeline(
    test_dir='dataset/test',
    output_dir='output',
    models_dir='models',
    exp_dir='experiments'
)

summary = pipeline.run_full_pipeline()
print("Inference completed:", summary)
```

## 5. Submission Validation

Run the official challenge validator to verify formatting compliance:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```
