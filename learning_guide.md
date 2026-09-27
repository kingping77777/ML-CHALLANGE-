# ML Challenge 2026: Business Entity Resolution Challenge - Learning Guide

This document summarizes the requirements, constraints, instructions, and our specific **3-Tier Implementation Strategy** for the ML Challenge 2026 based on the provided project files and our solution plan.

## 1. Problem Statement
The goal is to solve a **Business Entity Resolution (ER)** problem. You are provided with business records from three independent data sources (Source 1, Source 2, and Source 3). The data is noisy and inconsistent.
- **Task**: For every business entity in **Source 1 (the deduplicated reference source)**, find all matching records from **Source 2** and **Source 3**.
- A Source 1 entity may match zero, one, or many records in Sources 2 and 3.

## 2. Dataset Structure
- **Format**: Tab-separated values (`.tsv`). Always use `sep="\t"` when loading data.
- **Columns**:
  - `entity_id`: Unique ID for the record (prefixes S1-, S2-, or S3- indicate the source).
  - `business_name`: Name of the entity (includes typos, abbreviations, etc.).
  - `business_address`: Address of the entity (includes format variations, landmarks, missing components).
  - `country`: Country label (`US` and `India` in training, plus an unseen `France` in the test set).

### Directories:
- **`dataset/train/`**: Contains `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv` (labels).
  - Ground truth format: `source1_entity_id`, `matched_entity_ids` (comma-separated).
- **`dataset/test/`**: Contains `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` (generate matches for every Source 1 entity here).

## 3. Noise Patterns & Complexity Expected
- **Multilingual Name Variations**: Business names may appear in different languages, including English, French, and Hindi (Devanagari script vs. Romanized form). Standard ASCII normalization will destroy Devanagari characters, causing massive data loss.
- **Legal Suffixes & Name Variations**: "Pvt. Ltd.", "Private Limited", or simply "Pvt Ltd". Also expect DBA/trade names, ampersands (&) vs. "and", punctuation differences, word-order transpositions, and typos.
- **Address Formatting & Variations**: Inconsistent abbreviations (Rd vs Road, St vs Street), transliteration variants, missing components (no PIN code, no state), varying component orders, municipal numbering formats, and landmark-based references ("Near SBI ATM"). 
- **Complex Regional Addresses**: Indian addresses follow a complex, multi-part format that varies widely across cities and states (e.g., Door No, Cross, Main, Locality, City).
- **Unseen Geographies (Zero-Shot)**: The test set contains records from France, which has completely different address formats and language semantics not seen in the training data (US and India).
## 4. Output Requirements
Your solution must produce two TSV files in the `output/` folder:
1. **`matching_results.tsv` (Scored)**:
   - Columns: `source1_entity_id`, `matched_entity_ids` (comma-separated S2 and S3 IDs).
   - Singletons (no matches) should have an empty `matched_entity_ids`.
   - Must contain exactly one row for every Source 1 entity in the test set.
2. **`candidate_pairs.tsv` (Unscored, used for audit)**:
   - Columns: `source1_entity_id`, `candidate_entity_ids`.
   - Contains the candidate set generated from your blocking stage (before the final matching model).

*You can validate your files locally using the provided `utils/validate_submission.py` script before submitting.*

## 5. Evaluation Metric
Submissions are evaluated using the **Macro-Averaged F_0.5 Score**.
- **F_0.5** = `(1.25 × Precision × Recall) / (0.25 × Precision + Recall)`
- F_0.5 weights precision 2x over recall, penalizing false merges more than missed matches.
- Singletons correctly identified (empty match list) score 1.0; predicting matches for a true singleton scores 0.0.

## 6. Constraints and Rules
- External data lookup (APIs, databases, web scraping) is **STRICTLY PROHIBITED**. Use only the provided dataset.
- Final models must use an MIT/Apache 2.0 license and have up to 8 Billion parameters.
- No duplicate IDs in matched lists.
- You must create a `zip` submission package containing your `output/` files, runnable `code/`, and a completed `Documentation_template.md`.

---

## 7. Detailed 3-Tier Implementation Strategy

Based on competitive analysis, most teams will use simple TF-IDF blocking and flat XGBoost features. To achieve top performance, we will implement the following advanced pipeline:

### Step 1: Pre-Processing & Text Normalization
Before generating candidates, all text must be standardized to remove surface-level noise:
- **Legal Suffix Standardization:** Convert "Pvt. Ltd.", "Private Limited", etc., to a unified form like "pvt ltd".
- **Abbreviation & Symbol Replacement:** Convert ampersands ("&") to "and".
- **Punctuation Removal:** Strip dots, commas, and apostrophes.
- **Entity Extraction:** Extract ZIP/PIN codes explicitly from the raw address strings.

### Tier 1: Multi-Granularity Tiered Blocking (Candidate Generation)
Instead of comparing 2.2M entities against 5.3M records (11.7 trillion pairs), we will use parallel blocking strategies and route records based on data quality:
- **Filter 1 - Country Constraint:** Only compare US to US, India to India, and France to France (reduces space by ~60%).
- **Path A - Sparse Blocking:**
  - **Postal Code Index:** If two records share a ZIP/PIN, they are almost certainly in the same location.
  - **Word Index (TF-IDF):** Standard inverted index for names.
  - **Character Patterns (4-gram indexing):** Buckets typos (e.g., "mcdo" captures "McDonald", "Mcdonald's", "MC DONALDS").
- **Path B - Dense Neural Blocking (Crucial Gap):**
  - **Model:** `paraphrase-multilingual-MiniLM-L12-v2` (118M params, MIT license).
  - **Index:** FAISS HNSW Approximate Nearest Neighbor search.
  - **Why:** Handles French natively, captures semantic variations (e.g., "McDonald's Restaurant" vs "Mc Donalds"), and maps Indian Devanagari script to the same vector space as Romanized text, avoiding massive data loss from ASCII conversion.
- *Target:* Produce ~30 candidates per entity with >92% recall ceiling. Output saved to `candidate_pairs.tsv`.

### Tier 2: Machine Learning Classifier
For each candidate pair (approx. 50 per S1 entity), we compute ~17 features and feed them to an **XGBoost Classifier**:
- **Name Features:** Token Jaccard, Edit distance (Levenshtein), Jaro-Winkler, Sorted token match.
- **Address Features (Parsed):** Instead of a flat bag-of-words, parse addresses into `House number`, `Street`, `Locality`, and `City`. A house number mismatch is a strong negative signal. Include Token Jaccard on the full address.
- **Meta Features:** Postal code match, Country match, Source indicator (S2 vs S3), and **FAISS Cosine Similarity** from Tier 1.
- **Training Trick - Hard Negative Mining:** Train first on easy negatives (random pairs). Then, score candidates and collect pairs where the model falsely predicted high probability. Add these "hard negatives" to the training set and retrain so the model learns the exact confusion boundary.
- **Transformer Re-Ranker (P4):** Fine-tune a small `DistilBERT` model (66M params, Apache 2.0) to classify the top 5 candidates per entity (serialized as `"Name 1, Address 1 - Name 2, Address 2"`).

### Tier 3: Post-Processing & Optimization
Since F_0.5 heavily weights precision (2x over recall) and macro-averaging rewards singleton detection:
1. **Singleton Classifier Gate:** Detect the ~123,247 Source 1 entities that have zero true matches (5.6% of data). Features: max candidate score, score gap between top 2, candidate count, name rarity. These entities bypass the ML model and get an empty list, securing a 1.0 macro score contribution.
2. **One-to-One Bipartite Constraint:** Every S2 and S3 ID appears at most once in ground truth. If the model predicts the same S3 record for two different S1 entities, assign it only to the one with the highest probability.
3. **Threshold Sweeping:** Sweep the prediction threshold on a local validation set to strictly maximize Macro F_0.5.

### Prioritized Implementation Plan
- **P0**: Ensure all source files (including `train_source2.tsv` and `test_source3.tsv`) are downloaded and accessible.
- **P1**: Implement Dense FAISS blocking (multilingual embeddings) + One-to-one post-processing constraint.
- **P2**: Build the Singleton classifier gate + Add FAISS cosine similarity as an XGBoost feature.
- **P3**: Implement Hard negative mining curriculum + Address component parsing.
- **P4**: Add DistilBERT re-ranker on the top 5 candidates.
