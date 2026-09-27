# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** KingPing  
**Team Members:** Daksh  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

We developed a multi-stage Business Entity Resolution framework designed to identify duplicate entity records across disparate data sources (Source 1 vs. Source 2/3). Our solution integrates sparse inverted-index blocking with dense multilingual transformer FAISS embeddings, hard-negative mined XGBoost pairwise classification, a dedicated RandomForest singleton gate, and a strict 1-to-1 matching constraint optimized specifically for entity-level Macro $F_{0.5}$.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights from our initial EDA:
- High volume of singletons (entities with 0 true matches).
- Textual noise including legal suffixes (e.g. Inc, Corp, SARL, Pvt Ltd), address typos, script variations, and postal code formatting differences.
- Severe class imbalance (ratio of non-matches to true matches > 100:1).
- Uniqueness property: S2 and S3 candidate IDs appear at most once in ground truth matches.

### 2.2 Solution Strategy

**Approach Type:** Hybrid (Multi-tier Sparse + Dense FAISS Blocking + Hard-Negative XGBoost Classifier + Singleton Gate + 1-to-1 Post-processing)  
**Core Innovation:** A macro $F_{0.5}$-optimized post-processing chain that combines an entity-level aggregate singleton gate classifier with a deterministic 1-to-1 assignment solver, boosting validation macro $F_{0.5}$ from 0.9425 to **0.9646**.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.9\text{M}$ comparison space, we unified two complementary blocking streams:

1. **Sparse Blocking**: Inverted index over 4-grams, tokens, and country-aware postal codes with document frequency pruning ($df \le 1\%$).
2. **Dense FAISS Blocking**: `paraphrase-multilingual-MiniLM-L12-v2` embeddings indexed using FAISS HNSW ($M=32, efConstruction=200, efSearch=128$) to capture semantic and cross-lingual equivalence.
3. **Candidate Union (Tier-1)**: Combined candidates per Source 1 query entity up to $K=40$, preserving provenance.

---

## 4. Matching Model

**Features Used (40 Numerical Features):**
- **Name Features**: Normalized Levenshtein, Jaro-Winkler, Token Sort ratio, Token/4-gram Jaccard, token overlap ratios, prefix/suffix match indicators.
- **Address Features**: Normalized Levenshtein, Jaro-Winkler, Token overlap, Postal code exact/numeric match.
- **Structured Address Features**: Street number match, building unit match, city/state string similarity.
- **Multilingual Features**: Script mismatch indicator (Devanagari, Cyrillic, Latin), character set overlap.
- **Dense Similarity**: Cosine similarity from SentenceTransformer embeddings.
- **Metadata & Missingness**: Country match indicator, explicit missingness flags for name, address, and postal codes.

**Model Architecture:**
- **Pairwise Classifier**: Hard-Negative Mined XGBoost (trained with 2 iterations of false-positive mining).
- **Singleton Classifier**: Entity-level RandomForest trained on aggregated prediction probabilities ($\max, \text{mean}, \text{std}, \text{top}_1 - \text{top}_2$ prob differences).
- **Threshold Optimization**: Grid sweep over $[0.10, 0.99]$ on validation macro $F_{0.5}$, selecting **0.86** match threshold and **0.50** singleton threshold.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score:** **0.9646** (vs 0.9425 Baseline)
- **Validation Pairwise Precision:** **0.9782**
- **Validation Pairwise Recall:** **0.9214**
- **Singleton False Positive Rate:** **0.82%**
- **Observed Errors:**
  - *False Singletons*: Entities with very short generic names (e.g. "A & B Co") where sparse signals were ambiguous.
  - *False Positives*: Subsidiaries sharing identical corporate names and addresses differing only by minor suite numbers.

---

## 6. Conclusion

Our hybrid sparse-dense candidate union achieved high recall while maintaining a compact candidate pool ($K \le 40$). Coupled with hard-negative XGBoost training and our macro $F_{0.5}$-driven singleton gating & 1-to-1 constraint enforcement, the pipeline delivers state-of-the-art precision and recall for large-scale entity resolution.

---

## Appendix

### A. Code Artefacts
- Entry point for end-to-end test inference: `src/business_entity_resolution/pipeline.py`
- Self-contained code bundle: `code/business_entity_resolution/src/`
- Execution instructions: `code/business_entity_resolution/README.md`
- Dependencies: `code/business_entity_resolution/requirements.txt`
